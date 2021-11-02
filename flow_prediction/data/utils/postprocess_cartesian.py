import h5py
import numpy as np
import itertools
import concurrent.futures
import copy
import functools
import os
import scipy.interpolate
from tqdm import tqdm

from .mapping import get_vertices, get_maps
from .postprocess import _load_sensors, get_coordinates, get_field_variables, get_maps, get_sensor_locations, interpolate_to_sensors_and_full_flowfield

_real_dtypes = [np.float32, np.float64, np.float16, np.int64, np.int32, np.int16, np.int8]

def cartesian_grid_bounds(f, safety_fact = 0.95):
    ndims = f[list(f.keys())[0]]['coordinates'].shape[1]
    min_bounds = np.array([-np.inf for _ in range(ndims)])
    max_bounds = np.array([np.inf for _ in range(ndims)])
    for shape in f.keys():
        coords = np.array(f[shape]['coordinates'])
        max_coords = np.max(coords, axis=0)
        min_coords = np.min(coords, axis=0)
        min_bounds = np.max(np.stack([min_bounds, min_coords],0), axis=0)
        max_bounds = np.min(np.stack([max_bounds, max_coords],0), axis=0)
    centers = 0.5*(min_bounds + max_bounds)
    max_bounds_center_diff = max_bounds - centers
    min_bounds_center_diff = centers - min_bounds
    max_bounds = centers + safety_fact * max_bounds_center_diff
    min_bounds = centers - safety_fact * min_bounds_center_diff
    return min_bounds, max_bounds

def postprocess_cartesian(load_path, sensor_locations_z = None, sensor_placement_strategy_z = None, sensor_locations_w = None, sensor_placement_strategy_w = None, flowfield_sample_resolution = None, variables=None, mesh_folder = None, save_path=None, h5file_opts = None, verbose=False, nprocs = None, dt=1.0, grid_safety_fact = 0.80):
    '''
    Postprocess the raw pyfr simulation results to work with shallow decoder but with a cartesian target grid intead.

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
    -flowfield_sample_resolution: List of 2 ints. How many gridpoints to use in the x and y directions to sample the ground truth flowfield.
    -variables: List[Union[str, callable]]. Determines the quantities to extract from the raw flow variables. Default options are "p", "u", "v" and "vorticity". Any callable can be provided. Variable names will be recorded in the hdf5 from __name__ property.
    -mesh_folder: str. Directory containing mesh files. Mesh files are required for computing the S-C mappings.
    -save_path: str. Path to save the postprocessed data.
    -h5file_opts: Dict. kwargs to h5py.File.
    -verbose: bool. Prints progress information to stdout if True.
    -nprocs: int. Number of worker processes.
    -dt: float. Time step for computing time derivatives. If not supplied, a timestep of 1.0 will be assumed so you may have to manually divide the values by the actual dt later instead.
    '''
    
    assert load_path != save_path
    
    raw_data = h5py.File(load_path, 'r', **({} if h5file_opts is None else h5file_opts))

    mesh_folder = './' if mesh_folder is None else mesh_folder
    mesh_fnames = [mesh_folder + case + '.msh' for case in raw_data]

    soln_coords = get_coordinates(raw_data)
    variables = ['p','u','v'] if variables is None else variables
    soln_vals, variable_names = get_field_variables(raw_data, variables, dt=dt, verbose=verbose)
    if verbose:
        taskcounter = 1
        total_tasks = 6 if save_path is None else 7
        print(f'[{taskcounter}/{total_tasks}] Loaded solution coordinates and field variables {variables}')

    if (len(sensor_locations_z.shape) == 2) and (sensor_locations_z.shape[1] == 2) and (sensor_locations_z.dtype in _real_dtypes):
        sensor_locations_z = sensor_locations_z[:,0] + 1j*sensor_locations_z[:,1]
    #not doing this check for w as the inputs may be given in polar form

    grid_bounds = cartesian_grid_bounds(raw_data, safety_fact = grid_safety_fact)
    linspaces = [np.linspace(start,end,res) for start,end,res in zip(*grid_bounds, flowfield_sample_resolution)]
    full_flowfield_sensors = np.stack(np.meshgrid(*linspaces, indexing='ij'),-1).reshape(-1,2)
    full_flowfield_sensors = full_flowfield_sensors[:,0] + 1j * full_flowfield_sensors[:,1]
        
    sensor_locations_pfunc = functools.partial(get_sensor_locations, w_sensors = sensor_locations_w, z_sensors = sensor_locations_z, z_placement_modes = sensor_placement_strategy_z, w_placement_modes = sensor_placement_strategy_w)
    full_flowfield_locations_pfunc = functools.partial(get_sensor_locations, z_sensors = full_flowfield_sensors)
    
    vertices = get_vertices(mesh_fnames)
    if verbose:
        taskcounter += 1
        print(f'[{taskcounter}/{total_tasks}] Obtained object vertex coordinates')

    with concurrent.futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
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

            f.attrs['grid_shape'] = np.array(flowfield_sample_resolution, dtype=np.int64)
                
            for case, amap, slz, slw, ffz, ffw, (sensor_values,full_field_values) in zip(raw_data.keys(), amaps, sensor_locations_z, sensor_locations_w, full_flowfield_locations_z, full_flowfield_locations_w, interpolated_values):
                grp = f.create_group(case)
                grp.create_dataset('sensor_locations_z', data=slz)
                grp.create_dataset('sensor_locations_w', data=slw)
                grp.create_dataset('full_field_locations_z', data=ffz)
                grp.create_dataset('full_field_locations_w', data=ffw)
                grp.create_dataset('sensor_values', data = sensor_values)
                grp.create_dataset('full_field_values', data = full_field_values)
                grp_amap = grp.create_group('amap')
                amap.export_mapping_params_h5(grp_amap)
        if verbose:
            taskcounter += 1
            print(f"[{taskcounter}/{total_tasks}] Saved result")

    return sensor_locations_z, sensor_locations_w, full_flowfield_locations_z, full_flowfield_locations_w, interpolated_values

    
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
    parser.add_argument('--nprocs', type=int, default=None, help='Limit the number of workers. May be useful in memory startved situations.')
    parser.add_argument('--dt', type=float, default=1.0, help='Time step for computing time derivatives. If not supplied, a timestep of 1.0 will be assumed so you may have to manually divide the values by the actual dt later instead.')
    args = parser.parse_args()
    
    (z_sensor_strategy, z_sensor_coords), (w_sensor_strategy, w_sensor_coords) = _load_sensors(args.sensor_config)
    
    postprocess_cartesian(args.dataset,
                               save_path = args.output_path,
                               sensor_locations_z = z_sensor_coords,
                               sensor_placement_strategy_z = z_sensor_strategy,
                               sensor_locations_w = w_sensor_coords,
                               sensor_placement_strategy_w = w_sensor_strategy,
                               flowfield_sample_resolution = args.target_resolution,
                               variables = args.variables,
                               mesh_folder = args.mesh_folder,
                               verbose = args.verbose,
                               nprocs = args.nprocs,
                               dt = args.dt
                               )
