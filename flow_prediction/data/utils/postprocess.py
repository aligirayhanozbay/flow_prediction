import h5py
import numpy as np
import itertools
import concurrent.futures
import copy
import functools
import os
import scipy.interpolate

from .mapping import get_vertices, get_maps

_real_dtypes = [np.float32, np.float64, np.float16, np.int64, np.int32, np.int16, np.int8]

def generate_grid(amap, **kwargs):
    return amap._generate_annular_grid(**kwargs)

def _parse_z_placement_modes(placement_modes):
    if placement_modes is None:
        placement_modes = [None,None]
    elif isinstance(placement_modes, str):
        placement_modes = [placement_modes, placement_modes]
    else:
        placement_modes = copy.deepcopy(placement_modes)
        
    placement_map = {'behind': np.max, 'above': np.max, 'centered': np.mean, 'below': np.min, 'front': np.min, None: lambda x: 0.0}
    
    for k in range(len(placement_modes)):
        if not callable(placement_modes[k]):
            placement_modes[k] = placement_map[placement_modes[k]]
            
    return placement_modes

def _parse_w_placement_modes(placement_modes):
    if placement_modes is None:
        placement_modes = [None,None]
    elif isinstance(placement_modes, str):
        placement_modes = [placement_modes, placement_modes]
    else:
        placement_modes = copy.deepcopy(placement_modes)
        
    radial_modes = {'inner_relative': (lambda r,ri: r+ri), None: (lambda r,ri: r), 'outer_relative': (lambda r,ri: 1.0-np.abs(r)), 'scaled': (lambda r, ri: ri + (1-(1e-5)-ri) * r)}
    angular_modes = {None: (lambda theta, vertices: theta)}
    if not callable(placement_modes[0]):
        placement_modes[0] = radial_modes[placement_modes[0]]
    if not callable(placement_modes[1]):
        placement_modes[1] = angular_modes[placement_modes[1]]
    return placement_modes

def get_sensor_locations(annulusmap, z_sensors=None, w_sensors=None, z_placement_modes = None, w_placement_modes = None, forward_map_opts = None, backward_map_opts = None):
    
    vertices = annulusmap.mapping_params['inner_polygon_vertices']
    
    if z_sensors is not None:
        z_placement_modes = _parse_z_placement_modes(z_placement_modes)
        z_sensors = z_placement_modes[0](np.real(vertices)) + 1j*z_placement_modes[1](np.imag(vertices)) + z_sensors
        z_sensors_projected = annulusmap.backward_map(z_sensors)
    else:
        z_sensors = np.zeros((0,), dtype=np.complex128)
        z_sensors_projected = np.zeros((0,), dtype=np.complex128)

    if w_sensors is not None:
        if w_sensors.dtype in _real_dtypes:
            if len(w_sensors.shape) == 2 and w_sensors.shape[1] == 2:
                w_placement_modes = _parse_w_placement_modes(w_placement_modes)
                w_sensors[:,0] = w_placement_modes[0](w_sensors[:,0], annulusmap.mapping_params['inner_radius'])
                w_sensors[:,1] = w_placement_modes[1](w_sensors[:,1], vertices)
                w_sensors = w_sensors[:,0] * np.exp(1j*w_sensors[:,1]) #assuming they are of form (r,theta) for re^(i*theta)
            else:
                w_sensors = w_sensors.astype(np.complex128)
        w_sensors_projected = annulusmap.forward_map(w_sensors)
    else:
        w_sensors = np.zeros((0,), dtype=np.complex128)
        w_sensors_projected = np.zeros((0,), dtype=np.complex128)

    all_sensors_in_z_plane = np.concatenate([z_sensors, w_sensors_projected],0)
    all_sensors_in_w_plane = np.concatenate([z_sensors_projected, w_sensors],0)
        
    return all_sensors_in_z_plane, all_sensors_in_w_plane

def interpolate_to_sensors_and_full_flowfield(soln_vals, soln_coords, sensor_coords, full_flowfield_coords):
    #import pdb; pdb.set_trace()
    concatenated_coords_complex = np.concatenate([sensor_coords, full_flowfield_coords],0)
    concatenated_coords = np.stack([np.real(concatenated_coords_complex), np.imag(concatenated_coords_complex)], -1)

    interpolator = scipy.interpolate.CloughTocher2DInterpolator(soln_coords, soln_vals.transpose(1,0,2))
    target_vals = interpolator(concatenated_coords).transpose(1,0,2)

    return target_vals[:,:sensor_coords.shape[0]], target_vals[:,sensor_coords.shape[0]:,:]


class _field_var_computation:
    __name__ = None
    @staticmethod
    def __call__():
        raise(NotImplementedError(''))

class _get_field_var_p(_field_var_computation):
    __name__ ='p'
    @staticmethod
    def __call__(case):
        return case['solution'][...,0]

class _get_field_var_u(_field_var_computation):
    __name__ ='u'
    @staticmethod
    def __call__(case):
        return case['solution'][...,1]

class _get_field_var_v(_field_var_computation):
    __name__ ='v'
    @staticmethod
    def __call__(case):
        return case['solution'][...,2]

class _get_field_var_vorticity(_field_var_computation):
    __name__ ='vorticity'
    @staticmethod
    def __call__(case):
        return case['gradients'][...,4]-case['gradients'][...,3]
    

def get_field_variables(raw_data, variables):
    #variables: List[Union[str,callable]]. Names of variables. Currently s
    field_var_map = {'p': _get_field_var_p(), 'u': _get_field_var_u(), 'v': _get_field_var_v(), 'vorticity': _get_field_var_vorticity()}
    variables = copy.deepcopy(variables)
    for k in range(len(variables)):
        if not callable(variables[k]):
            variables[k] = field_var_map[variables[k]]

    field_vals = []
    for case in raw_data:
         field_val = np.stack([get_var(raw_data[case]) for get_var in variables],-1)
         field_vals.append(field_val)

    variable_names = [v.__name__ for v in variables]

    return field_vals, variable_names

def get_coordinates(raw_data):
    return [np.array(raw_data[case]['coordinates']) for case in raw_data]

def postprocess_shallowdecoder(load_path, sensor_locations_z = None, sensor_placement_strategy_z = None, sensor_locations_w = None, sensor_placement_strategy_w = None, flowfield_sample_resolution = None, variables=None, mesh_folder = None, save_path=None, h5file_opts = None, verbose=False):
    '''
    Postprocess the raw pyfr simulation results to work with shallow decoder training.

    Inputs:
    -load_path: str. Path to the .h5 file containing the pyfr solutions.
    -sensor_locations_z: np.array[Union[np.complex128, np.float64]]. Locations of the sparse sensors in the z-plane. Can either be a 1d array with complex values, or a 2d array of shape (n_sensors,2).
    -sensor_placement_strategy_z: List[str]. How the sensors should be placed in the x- and y- directions, relative to the object vertices in the domain. First element determines the strategy in the x-dir and the second in the y-dir. Options are 'behind', 'above', 'centered', 'below', 'front', None. None takes the raw values.
    -sensor_locations_w: Similar to sensor_locations_z, but in the annular domain. If a 2d input with shape (n_sensors,2) is given, this is interpreted as polar coordinates.
    -sensor_placement_strategy_w: Similar to sensor_placement_strategy_z, but for the annular domain. Options are: 
        'inner_relative': Chooses radial coordinate as the inner ring radius + given values
        'outer_relative': Chooses radial coordinate as the outer ring radius - given values
        'scaled': Expects values between 0.0 and 1.0. 0.0 corresponds to the inner ring and 1.0 to the outer.
        None: Use raw values.
    -flowfield_sample_resolution: List of 2 ints. How many gridpoints to use in the radial and angular directions to sample the ground truth flowfield.
    -variables: List[Union[str, callable]]. Determines the quantities to extract from the raw flow variables. Default options are "p", "u", "v" and "vorticity". Any callable can be provided. Variable names will be recorded in the hdf5 from __name__ property.
    -mesh_folder: str. Directory containing mesh files. Mesh files are required for computing the S-C mappings.
    -save_path: str. Path to save the postprocessed data.
    -h5file_opts: Dict. kwargs to h5py.File.
    -verbose: bool. Prints progress information to stdout if True.
    '''
    
    assert load_path != save_path
    
    raw_data = h5py.File(load_path, 'r', **({} if h5file_opts is None else h5file_opts))

    mesh_folder = './' if mesh_folder is None else mesh_folder
    mesh_fnames = [mesh_folder + case + '.msh' for case in raw_data]

    soln_coords = get_coordinates(raw_data)
    variables = ['p','u','v'] if variables is None else variables
    soln_vals, variable_names = get_field_variables(raw_data, variables)
    if verbose:
        taskcounter = 1
        total_tasks = 6 if save_path is None else 7
        print(f'[{taskcounter}/{total_tasks}] Loaded solution coordinates and field variables {variables}')

    if (len(sensor_locations_z.shape) == 2) and (sensor_locations_z.shape[1] == 2) and (sensor_locations_z.dtype in _real_dtypes):
        sensor_locations_z = sensor_locations_z[:,0] + 1j*sensor_locations_z[:,1]
    #not doing this check for w as the inputs may be given in polar form

    full_flowfield_sensors = np.stack(np.meshgrid(np.linspace(0.0,0.96,flowfield_sample_resolution[0]), np.linspace(0.0,2*np.pi,flowfield_sample_resolution[1]), indexing='ij'),-1).reshape(-1,2)
        
    sensor_locations_pfunc = functools.partial(get_sensor_locations, w_sensors = sensor_locations_w, z_sensors = sensor_locations_z, z_placement_modes = sensor_placement_strategy_z, w_placement_modes = sensor_placement_strategy_w)
    full_flowfield_locations_pfunc = functools.partial(get_sensor_locations, w_sensors = full_flowfield_sensors, w_placement_modes = ['scaled', None])

    vertices = get_vertices(mesh_fnames)
    if verbose:
        taskcounter += 1
        print(f'[{taskcounter}/{total_tasks}] Obtained object vertex coordinates')
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        amaps = list(executor.map(get_maps, vertices))
        if verbose:
            taskcounter += 1
            print(f'[{taskcounter}/{total_tasks}] Computed conformal mappings')
        
        sensor_locations = list(executor.map(sensor_locations_pfunc, amaps))
        if verbose:
            taskcounter += 1
            print(f"[{taskcounter}/{total_tasks}] Computed sensors' locations in original and transformed coordinates")
            
        full_flowfield_locations = list(executor.map(full_flowfield_locations_pfunc, amaps))
        if verbose:
            taskcounter += 1
            print(f"[{taskcounter}/{total_tasks}] Computed dense mesh point locations in original and transformed coordinates")
            
        sensor_locations_z, sensor_locations_w = list(zip(*sensor_locations))
        full_flowfield_locations_z, full_flowfield_locations_w = list(zip(*full_flowfield_locations))
        
        interpolated_values = list(executor.map(interpolate_to_sensors_and_full_flowfield, soln_vals, soln_coords, sensor_locations_z, full_flowfield_locations_z))
        if verbose:
            taskcounter += 1
            print(f"[{taskcounter}/{total_tasks}] Interpolated field variables to sensor and dense meshes")

    if save_path is not None:
        with h5py.File(save_path, 'w') as f:
            for k,varname in enumerate(variable_names):
                f.attrs['variable' + str(k)] = varname.encode('utf-8')
                
            for case, slz, slw, ffz, ffw, (sensor_values,full_field_values) in zip(raw_data.keys(), sensor_locations_z, sensor_locations_w, full_flowfield_locations_z, full_flowfield_locations_w, interpolated_values):
                grp = f.create_group(case)
                grp.create_dataset('sensor_locations_z', data=slz)
                grp.create_dataset('sensor_locations_w', data=slw)
                grp.create_dataset('full_field_locations_z', data=ffz)
                grp.create_dataset('full_field_locations_w', data=ffw)
                grp.create_dataset('sensor_values', data = sensor_values)
                grp.create_dataset('full_field_values', data = full_field_values)
        if verbose:
            taskcounter += 1
            print(f"[{taskcounter}/{total_tasks}] Saved result")

    return sensor_locations_z, sensor_locations_w, full_flowfield_locations_z, full_flowfield_locations_w, interpolated_values


def _load_sensor_coords(v):
    if v is None:
        return None
    elif isinstance(v, str):
        v = np.load(v)
    else:
        v = np.asarray(v)
    assert ((len(v.shape) == 1) or ((len(v.shape) == 2) and (v.shape[1] == 2)))
    return v  

def _load_sensors(cfg_path):
    if os.path.splitext(cfg_path)[1] == '.json':
        import json
        sensor_config = json.load(open(cfg_path, 'r'))
    else:
        import pickle
        sensor_config = pickle.load(open(cfg_path,'rb'))

    strategies = [sensor_config.get(x,None) for x in ['z_strategy', 'w_strategy']]
    coords = [_load_sensor_coords(sensor_config.get(x,None)) for x in ['z_coordinates', 'w_coordinates']]
    
    return (strategies[0], coords[0]), (strategies[1], coords[1])

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help = 'Path to the .h5 file containing the pyfr solutions')
    parser.add_argument('sensor_config', type=str, help = "A json or pickle file containing a dict, with the fields 'z_strategy', 'w_strategy', 'z_coordinates', 'w_coordinates'. See signature of postprocess_shallowdecoder.")
    parser.add_argument('output_path', type=str, help = 'Path to save the postprocessed data')
    parser.add_argument('--target-resolution', nargs = 2, type=int, default=(50,200), help = '# of gridpoints in the radial and angular dimensions on the annular domains, for getting the full flowfield samples')
    parser.add_argument('--mesh_folder', type=str, default='./', help = 'Directory containing mesh files. Mesh files are required for computing the S-C mappings.')
    parser.add_argument('--variables', nargs='*', default=['p','u','v'], help = 'Names of the variables to extract from the .h5 file. Current options are: p, u, v, vorticity')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    (z_sensor_strategy, z_sensor_coords), (w_sensor_strategy, w_sensor_coords) = _load_sensors(args.sensor_config)
    
    postprocess_shallowdecoder(args.dataset,
                               save_path = args.output_path,
                               sensor_locations_z = z_sensor_coords,
                               sensor_placement_strategy_z = z_sensor_strategy,
                               sensor_locations_w = w_sensor_coords,
                               sensor_placement_strategy_w = w_sensor_strategy,
                               flowfield_sample_resolution = args.target_resolution,
                               variables = args.variables,
                               mesh_folder = args.mesh_folder,
                               verbose = args.verbose
                               )
