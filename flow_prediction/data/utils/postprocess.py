import h5py
import numpy as np
import itertools
import concurrent.futures
import copy
import functools
import scipy.interpolate

from test_maps import get_vertices, get_maps

def rewrite_to_disk(data, target_path):
    #pointless code to copy h5py file essentially. dont actually use.
    newfile = h5py.File(target_path, 'w')
    for groupname in data:
        newfile.create_group(groupname)
        for dataname in data[groupname]:
            newfile[groupname].create_dataset(dataname, data=data[groupname][dataname])
            print(f'Written {groupname}/{dataname} to {target_path}')
    newfile.close()
    
def _load_single_subgroup(h5path, subgroup, verbose = False):
    subgroup_as_dict = {}
    with h5py.File(h5path, 'r') as f:
        for k in f[subgroup].keys():
            if verbose:
                print(f'Reading {k} in {subgroup} from {h5path}')
            subgroup_as_dict[k] = np.array(f[subgroup][k])
        #subgroup_as_dict = {k:np.array(f[subgroup][k]) for k in f[subgroup].keys()}
    return subgroup_as_dict

def load_to_ram(h5path):
    f = h5py.File(h5path, 'r')
    subgroup_names = list(f.keys())
    f.close()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        subgroup_values = list(executor.map(_load_single_subgroup, itertools.repeat(h5path, len(subgroup_names)), subgroup_names))
    return {k:v for k,v in zip(subgroup_names, subgroup_values)}

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
    real_dtypes = [np.float32, np.float64, np.float16, np.int64, np.int32, np.int16, np.int8]
    vertices = annulusmap.mapping_params['inner_polygon_vertices']
    # if isinstance(z_sensors,np.ndarray) and len(z_sensors.shape) == 2 and z_sensors.shape[1] == 2 and (z_sensors.dtype in real_dtypes):
    #     z_sensors = z_sensors[:,0] + 1j*z_sensors[:,1]
    # import pdb; pdb.set_trace()
    if z_sensors is not None:
        z_placement_modes = _parse_z_placement_modes(z_placement_modes)
        z_sensors = z_placement_modes[0](np.real(vertices)) + z_placement_modes[1](np.imag(vertices)) + z_sensors
        z_sensors_projected = annulusmap.backward_map(z_sensors)
    else:
        z_sensors = np.zeros((0,), dtype=np.complex128)
        z_sensors_projected = np.zeros((0,), dtype=np.complex128)

    if w_sensors is not None:
        if w_sensors.dtype in real_dtypes:
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
    
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--target-resolution', nargs = 2, type=int, default=(50,200))
    parser.add_argument('--mesh_folder', type=str, default='./')
    args = parser.parse_args()

    assert args.dataset != args.output_path
    
    import time
    t0 = time.time()
    raw_data = load_to_ram(args.dataset)
    t1 = time.time()
    ttaken = t1-t0
    print(f'Loaded {args.dataset} in {ttaken} seconds')
    mesh_fnames = [args.mesh_folder + case + '.msh' for case in raw_data]
    soln_coords = [raw_data[case]['coordinates'] for case in raw_data]
    soln_vals = [raw_data[case]['solution'] for case in raw_data]
    

    vel_sensors = np.array([0.15+0.1*1j, 0.30+0.1*1j, 0.15-0.1*1j, 0.3-0.1*1j])
    vel_sensors_strategy = ['behind', 'centered']

    n_pressure_sensors = 10
    p_sensors = np.stack([np.zeros((n_pressure_sensors,)), np.linspace(0.0,2*np.pi,n_pressure_sensors)],-1)
    p_sensors_strategy = ['inner_relative', None]

    full_flowfield_sensors = np.stack(np.meshgrid(np.linspace(0.0,0.96,args.target_resolution[0]), np.linspace(0.0,2*np.pi,args.target_resolution[1]), indexing='ij'),-1).reshape(-1,2)

    n_sensors = p_sensors.shape[0] + vel_sensors.shape[0]

    sensor_locations_pfunc = functools.partial(get_sensor_locations, w_sensors = p_sensors, z_sensors = vel_sensors, z_placement_modes = vel_sensors_strategy, w_placement_modes = p_sensors_strategy)
    full_flowfield_locations_pfunc = functools.partial(get_sensor_locations, w_sensors = full_flowfield_sensors, w_placement_modes = ['scaled', None])
    
    
    vertices = get_vertices(mesh_fnames)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        amaps = list(executor.map(get_maps, vertices))
        sensor_locations = list(executor.map(sensor_locations_pfunc, amaps))
        full_flowfield_locations = list(executor.map(full_flowfield_locations_pfunc, amaps))
        sensor_locations_z, sensor_locations_w = list(zip(*sensor_locations))
        full_flowfield_locations_z, full_flowfield_locations_w = list(zip(*full_flowfield_locations))
        interpolated_values = list(executor.map(interpolate_to_sensors_and_full_flowfield, soln_vals, soln_coords, sensor_locations_z, full_flowfield_locations_z))
    # interpolated_values = list(map(interpolate_to_sensors_and_full_flowfield, soln_vals, soln_coords, sensor_locations_z, full_flowfield_locations_z))
    #sensor_locations = list(map(sensor_locations_pfunc, amaps))

    with h5py.File(args.output_path, 'w') as f:
        for case, slz, slw, ffz, ffw, (sensor_values,full_field_values) in zip(raw_data.keys(), sensor_locations_z, sensor_locations_w, full_flowfield_locations_z, full_flowfield_locations_w, interpolated_values):
            grp = f.create_group(case)
            grp.create_dataset('sensor_locations_z', data=slz)
            grp.create_dataset('sensor_locations_w', data=slw)
            grp.create_dataset('full_field_locations_z', data=ffz)
            grp.create_dataset('full_field_locations_w', data=ffw)
            grp.create_dataset('sensor_values', data = sensor_values)
            grp.create_dataset('full_field_values', data = full_field_values)
    
