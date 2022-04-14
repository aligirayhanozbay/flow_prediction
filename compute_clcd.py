import argparse
import functools
import itertools
import json
import copy
import pickle
from tqdm import tqdm
import h5py
import tensorflow as tf
import numpy as np
import pydscpack
import scipy.spatial, scipy.interpolate

from spatial_plotting import extract_dataset_metadata, _init_models as _init_models_base, _init_dataset as _init_dataset_base, mape_with_threshold as mape
from spatiotemporal_plotting import _init_dataset as _init_dataset_spatiotemporal, _init_models as _init_models_spatiotemporal

def _parse_args_clcd():

    parser = argparse.ArgumentParser('Compute ground truth and predicted Cl/Cd values')

    parser.add_argument('dataset', type=str, help='path to the dataset file')
    parser.add_argument('-m', '--model',
                        action='append',
                        nargs=4,
                        metavar=('identifier', 'model_type', 'config', 'weights')
                        )
    parser.add_argument('-r', '--reconstruction_model',
                        nargs=3,
                        metavar=('identifier', 'model_type', 'weights'),
                        action='append',
                        help='model used to reconstruct the snapshots at t. supply exactly as many as models supplied via -m. args.reconstruction_model[i] is used to generate inputs for args.model[i]. do not supply any for spatial-only/non-denoised results.',
                        default=None)
    parser.add_argument('-b', type=int, help='batch size override', default=None)
    parser.add_argument('-o', type=str, help='output file path. if not provided, results will be printed instead.', default=None)
    parser.add_argument('-i', action='store_true', help='interpolate field values to object surfaces. useful for e.g. cartesian sampling datasets.')
    parser.add_argument('--rho', type=float, help='Density', default=1.0)
    parser.add_argument('--nu', type=float, help='1/(Reynolds number)', default=1.0)
    
    
    args = parser.parse_args()

    return args


def _init_models(x, input_shapes, output_shapes, var_order=None):
    
    if input_shapes is None:
        os = []
        for o in output_shapes:
            if len(o) == 2:
                o = [1,1] + list(o)
            elif len(o) == 3:
                o = [None] + list(o)
            os.append(o)
        models = list(itertools.chain.from_iterable(list(map(
            lambda m, *args: _init_models_spatiotemporal([m], *args),
            x, os
        ))))
    else:
        models = list(itertools.chain.from_iterable(list(map(
            lambda m,*args: _init_models_base([m],*args),
            x, input_shapes, output_shapes
        ))))

    #field map tells us which model to use for each variable
    #if two models predict the same variable, this function prioritizes the first
    required_fields = set(['u','v','p'])
    field_map = {}
    for k, (_, _, config, _) in enumerate(x):
        config = json.load(open(config,'r'))
        full_field_mask = config['dataset'].get('full_field_mask', var_order)
        f = set(full_field_mask).intersection(required_fields)
        for field in f:
            field_map[field] = (k,full_field_mask.index(field))
        required_fields = required_fields - f

    return models, field_map

def _dset_map_variable(nvars, var_indices, x, y, *args, dense=False):
    if dense:
        yshape = tf.shape(y)
        newshape = tf.concat([yshape[:-1],[-1,nvars]],0)
        yn = tf.reshape(y,newshape)
        yn = tf.experimental.numpy.swapaxes(yn,0,-1)
        yn = tf.gather(yn, var_indices)
        yn = tf.experimental.numpy.swapaxes(yn,0,-1)
        yn = tf.reshape(yn, tf.concat([yshape[:-1],[-1]],0))   
    else:
        channel_axis = -1 if (tf.keras.backend.image_data_format() == 'channels_last') else 1
        yn = tf.experimental.numpy.swapaxes(y,0,channel_axis)
        yn = tf.gather(yn, var_indices)
        yn = tf.experimental.numpy.swapaxes(yn,0,channel_axis)

    return x, yn, *args

def _unfiltered_var_info(dataset_path):
    with h5py.File(dataset_path, 'r') as f:
        nvars = len(list(f.attrs.keys()))-1
        return nvars, [f.attrs[f'variable{k}'] for k in range(nvars)]

def _init_dataset(path, configs, bsize=None, use_training_dataset=False, reconstruction_models=None):
    
    dset_configs = []
    for config in configs:
        if isinstance(config, str):
            dset_config = json.load(open(config, 'r'))['dataset']
        else:
            dset_config = copy.deepcopy(config)
        dset_configs.append(dset_config)
    
    if reconstruction_models is not None:
        model_dsets = []
        for config,rm in zip(dset_configs, reconstruction_models):
            rm = [rm[1],rm[0],rm[2]]
            ds = _init_dataset_spatiotemporal(path,
                                              config,
                                              reconstruction_model=rm,
                                              bsize=bsize)[int(not use_training_dataset)]
            model_dsets.append(ds)
        base_ds = None
    else:
        metadatas = list(map(extract_dataset_metadata, itertools.repeat(path), [d.get('full_field_mask',None) for d in dset_configs]))
        base_config = copy.deepcopy(dset_configs[0])
        _ = base_config.pop('full_field_mask',None)
        base_ds = _init_dataset_base(path, base_config, bsize=bsize, reshape_to_grid=True, retain_variable_dimension=True)[int(not use_training_dataset)]

        model_dsets = []
        nvars, var_order = _unfiltered_var_info(path)
        for md,config in zip(metadatas,dset_configs):
            ffm = config.get('full_field_mask',None)
            unfiltered_indices = [var_order.index(v) for v in ffm]
            pf = functools.partial(_dset_map_variable, nvars, unfiltered_indices)
            if ffm is None:
                model_dsets.append(base_ds)
            else:
                dset = base_ds.map(pf)
                model_dsets.append(dset)

    return base_ds, model_dsets

def predict(models, datasets, identifiers=None):

    print('Running eval...')
    if identifiers is None:
        identifiers = list(range(len(models)))

    overall_results = []
    for model,identifier,dataset in zip(models,identifiers,datasets):
        is_spatiotemporal_dataset = not isinstance(dataset, tf.data.Dataset)
        n_batches = len(dataset) if is_spatiotemporal_dataset else int(dataset.cardinality())
        
        inputs = []
        norm_params = []
        case_names = []
        ground_truths = []
        preds = []
        
        pbar = tqdm(total = n_batches)
        pbar.set_description(str(identifier))
        
        for data in iter(dataset):
            if is_spatiotemporal_dataset:
                x,y,r,n,c = data
                preds.append(model(r[:,0]).numpy())
                case_names.append(c[:,0])
                ground_truths.append(y)
                norm_params.append(n[:,1])
                inputs.append(x)
            else:
                x,y,n,c = data
                preds.append(model(x).numpy())
                case_names.append(np.array(list(map(lambda x: x.decode('utf8'),c.numpy()))))
                ground_truths.append(y.numpy())
                norm_params.append(n.numpy())
                inputs.append(x.numpy())
            pbar.update(1)
        results = {
            'ground_truths': np.concatenate(ground_truths,0),
            'case_names': np.concatenate(case_names,0),
            'norm_params': np.concatenate(norm_params,0),
            'inputs': np.concatenate(inputs,0),
            'predictions': np.concatenate(preds,0)
        }
        overall_results.append(results)
        
    return overall_results

def interpolate_to(c,values,t):
    res = []
    for v in values:
        interpolator = scipy.interpolate.CloughTocher2DInterpolator(c,v.reshape(-1))
        res.append(interpolator(t))
    return np.stack(res,0)

def imag_to_2d(z):
    return np.stack([np.real(z), np.imag(z)],-1)

def compute_clcd(results, field_map, dset_path, dataset_metadata, rho, nu, interpolate=False):
    
    p_preds = results[field_map['p'][0]]['predictions'][:,field_map['p'][1]] + np.expand_dims(results[field_map['p'][0]]['norm_params'][:,field_map['p'][1]], [1,2])
    u_preds = results[field_map['u'][0]]['predictions'][:,field_map['u'][1]] + np.expand_dims(results[field_map['u'][0]]['norm_params'][:,field_map['u'][1]], [1,2])
    v_preds = results[field_map['v'][0]]['predictions'][:,field_map['v'][1]] + np.expand_dims(results[field_map['v'][0]]['norm_params'][:,field_map['v'][1]], [1,2])

    p_gt = results[field_map['p'][0]]['ground_truths'][:,field_map['p'][1]] + np.expand_dims(results[field_map['p'][0]]['norm_params'][:,field_map['p'][1]], [1,2])
    u_gt = results[field_map['u'][0]]['ground_truths'][:,field_map['u'][1]] + np.expand_dims(results[field_map['u'][0]]['norm_params'][:,field_map['u'][1]], [1,2])
    v_gt = results[field_map['v'][0]]['ground_truths'][:,field_map['v'][1]] + np.expand_dims(results[field_map['v'][0]]['norm_params'][:,field_map['v'][1]], [1,2])
    
    dset = h5py.File(dset_path,'r')

    forces = {}
    
    for shape in tqdm(dset, desc='Computing Cl and Cd'):
        mp = {k:np.array(dset[shape]['amap'][k]) for k in dset[shape]['amap']}
        amap = pydscpack.AnnulusMap(mapping_params = mp)

        indices = np.where(results[0]['case_names'] == shape)[0]

        if len(indices) == 0:
            continue

        #do interpolation if needed
        if interpolate:
            #get coordinates of near field gridpts
            grid_coords = imag_to_2d(dset[shape]['full_field_locations_z'] )
            tess = scipy.spatial.Delaunay(grid_coords)

            near_field_w, near_field_z = map(lambda x: x[:2], amap._generate_annular_grid(n_pts=interpolate if (isinstance(interpolate, tuple) or isinstance(interpolate, list)) else (64,256)))
            w = near_field_w[1]
            target_coords = imag_to_2d(near_field_z)

            #get surface field values
            surface_pressure_pred = interpolate_to(tess,p_preds[indices],target_coords[0])
            surface_pressure_gt = interpolate_to(tess,p_gt[indices], target_coords[0])

            surface_u_pred = interpolate_to(tess, u_preds[indices], target_coords[1])
            surface_u_gt = interpolate_to(tess, u_gt[indices], target_coords[1])
            
            surface_v_pred = interpolate_to(tess, v_preds[indices], target_coords[1])
            surface_v_gt = interpolate_to(tess, v_gt[indices], target_coords[1])
            
        else:
            #get coordinates of near field gridpts
            near_field_z = np.array(dset[shape]['full_field_locations_z']).reshape(*dataset_metadata['grid_shape'])[:2,:]
            w = np.array(dset[shape]['full_field_locations_w']).reshape(*dataset_metadata['grid_shape'])[1,:]

            #get surface field values
            surface_pressure_pred = p_preds[indices,0,:]
            surface_pressure_gt = p_gt[indices,0,:]

            surface_u_pred = u_preds[indices,1,:]
            surface_u_gt = u_gt[indices,1,:]

            surface_v_pred = v_preds[indices,1,:]
            surface_v_gt = v_gt[indices,1,:]

        #get surface normals
        #[0,:]->surface of the object
        surface_z, n1 = near_field_z
        surface_z_padded = np.pad(surface_z, (1,1), mode='wrap')
        surface_tangent = surface_z_padded[2:] - surface_z_padded[:-2]
        surface_tangent = surface_tangent / np.sqrt(surface_tangent * np.conj(surface_tangent))
        surface_normal = surface_tangent * (-1j)

        #pressure force
        pp = (-surface_normal) * surface_pressure_pred
        pg = (-surface_normal) * surface_pressure_gt
        
        #need to compute vel. vector parallel to the object surface just above the surface, u[1,:] and v[1,:]
        #let r(t)=x(t)+i*y(t)=(t+r_i)*exp(i*theta) be a line from the inner ring of the annulus to the outer ring
        #the tangents of this line are found by dr/dt. let f(w) be the conformal transformation for the shape
        #to find the tangents in the z-domain we need to simply do (df/dz)*(dr/dt)
        #rotate tangent for the normal. as dt->0, the normal will be parallel to the object surface.
        #assume velocities on the object surface is 0. 
        drdt = np.exp(1j*np.angle(w))
        dzdw = amap.dzdw(w)
        radial_tangent = dzdw*drdt
        radial_tangent = radial_tangent/((radial_tangent*np.conj(radial_tangent))**0.5)
        radial_normal = radial_tangent * (1j)
        rot_angle = np.angle(radial_normal)
        rot_matrices = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],[np.sin(rot_angle), np.cos(rot_angle)]])

        rotated_velocities_parallel_pred = np.einsum('ij...,j...->i...',
                                                     rot_matrices,
                                                     np.stack([surface_u_pred, surface_v_pred],0))[0]
        rotated_velocities_parallel_gt = np.einsum('ij...,j...->i...',
                                                   rot_matrices,
                                                   np.stack([surface_u_gt, surface_v_gt],0))[0]
        dn = n1-surface_z
        dn = np.real((dn*np.conj(dn))**0.5)

        shear_stresses_pred = nu*rho*(rotated_velocities_parallel_pred / dn)*surface_tangent
        shear_stresses_gt = nu*rho*(rotated_velocities_parallel_gt / dn)*surface_tangent

        #pressure force plus the shear stress at a point will be the force at that point
        f_p = pp + shear_stresses_pred
        f_g = pg + shear_stresses_gt

        #get arc lengths
        ds = surface_z[1:] - surface_z[:-1]
        ds = np.real((ds * np.conj(ds))**0.5)
        s = np.concatenate([[0],np.cumsum(ds)],0)

        #integrate using trapezoidal rule
        Fp = np.trapz(f_p, s)
        Fg = np.trapz(f_g, s)

        #Decompose to x and y forces (drag and lift)
        Fxp = np.real(Fp)
        Fyp = np.imag(Fp)
        Fxg = np.real(Fg)
        Fyg = np.imag(Fg)

        forces[shape] = {
            "predicted": {"x": Fxp, "y": Fyp},
            "target": {"x": Fxg, "y": Fyg}
        }
        
    forces_pct_error = {
        'x': mape(
            np.stack([forces[shape]['predicted']['x'] for shape in forces],0),
            np.stack([forces[shape]['target']['x'] for shape in forces],0),
            pcterror_threshold = 200
        ),
        'y': mape(
            np.stack([forces[shape]['predicted']['y'] for shape in forces],0),
            np.stack([forces[shape]['target']['y'] for shape in forces],0),
            pcterror_threshold = 200
        )
    }

    mae = lambda yp,yt: tf.reduce_mean(tf.abs(yp-yt))
    forces_abs_error = {
        'x': mae(
            np.stack([forces[shape]['predicted']['x'] for shape in forces],0),
            np.stack([forces[shape]['target']['x'] for shape in forces],0)
        ),
        'y': mae(
            np.stack([forces[shape]['predicted']['y'] for shape in forces],0),
            np.stack([forces[shape]['target']['y'] for shape in forces],0)
        )
    }

    return forces, forces_pct_error, forces_abs_error

if __name__ == '__main__':

    tf.keras.backend.set_image_data_format('channels_first')
    args = _parse_args_clcd()

    base_ds, model_ds = _init_dataset(args.dataset,
                                      [m[2] for m in args.model],
                                      reconstruction_models=args.reconstruction_model,
                                      bsize=args.b)
    
    if args.reconstruction_model is None:
        models, field_map = _init_models(args.model,
                                         [m.element_spec[0].shape for m in model_ds],
                                         [m.element_spec[1].shape for m in model_ds],
                                         var_order=_unfiltered_var_info(args.dataset)[1])
    else:
        models, field_map = _init_models(args.model,
                                         None,
                                         [m.data_shape for m in model_ds],
                                         var_order=_unfiltered_var_info(args.dataset)[1])

    ffm = json.load(open(args.model[0][2],'r'))['dataset'].get('full_field_mask',None)
    dataset_metadata = extract_dataset_metadata(args.dataset,ffm)

    preds = predict(models,model_ds,[x[0] for x in args.model])

    clcd = compute_clcd(preds, field_map, args.dataset, dataset_metadata, args.rho, args.nu, interpolate=args.i)

    res = {
        'forces': clcd[0],
        'forces_pct_error': clcd[1],
        'forces_abs_error': clcd[2]
    }

    if args.o is not None:
        pickle.dump(res, open(args.o,'wb'))
    else:
        print(res)

    
    
    
