import json
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import numpy as np
import matplotlib.path as mplPath
import argparse
import re
import copy
import itertools
from tqdm import tqdm

from flow_prediction.models.SD_UNet import SD_UNet
from flow_prediction.models.SD_FNO import SD_FNO
from flow_prediction.models.FNO import FNO
from flow_prediction.data.training.TimeMarchingDataset import TimeMarchingDataset
from flow_prediction.data.training.ShallowDecoderDataset import _extract_variable_names, _get_unique_variable_mask_ids

from spatial_plotting import extract_dataset_metadata, gather_masked_indices, check_in_polygon, mape_with_threshold
from spatial_plotting import _init_models as _init_reconstruction_model

def _parse_args_spatiotemporal():

    parser=argparse.ArgumentParser('Generate plots for the spatial-only case')
    
    parser.add_argument('dataset', type=str, help='path to the dataset file')
    parser.add_argument('-o', type=str, default='./', help='output directory')
    parser.add_argument('-s', nargs='*', type=int, help='indices of snapshots to use', default=[0])
    parser.add_argument('-c', nargs='*', type=float, help='provide a value for each snapshot, or a single value for all snapshots. clips the colorbars in the plots. e.g. supplying 0.075 for a snapshot means that vmax=-vmin=0.075*max(abs(values in the snapshot)). default is 0.075.', default=[0.075])
    parser.add_argument('-l', action='store_false', help='interpret colorbar extents as literal values instead of a % of max(abs(values in the snapshot))')
    parser.add_argument('-p', nargs=2, type=float, help='colobar limits for the % error plot', default=[0,100])
    parser.add_argument('-r', '--reconstruction-model',
                        nargs=3,
                        metavar=('identifier', 'model_type', 'weights'),
                        help='model used to reconstruct the snapshots at t')
    parser.add_argument('-m','--model',
                        action='append',
                        nargs=4,
                        metavar=('identifier', 'model_type', 'config', 'weights'))
    parser.add_argument('-e', nargs=4, type=float, help='extents of the plotting domain',
                        metavar=('xmin', 'xmax', 'ymin', 'ymax'),
                        default=[-1.5,6.0,-1.5,1.5])
    parser.add_argument('--prefix', type=str, default=None, help='prefix for all output files')
    parser.add_argument('--traindata', action='store_true', help='use training data instead of test data')
    parser.add_argument('--pcterrt', type=float, help='% errors above this value will be discarded. can be used to filter out extremely high % errors caused due very small denominators.', default=np.inf)
    parser.add_argument('--magnitudet', type=float, help='float between 0 and 1. if set to e.g. 0.05, for some field f, if a ground truth data pt has absolute value equal to or smaller than 5% of max(|f|), that point is excluded from error metrics', default=0.0)
    parser.add_argument('--bsize', type=int, help='override the batch size used by the dataset', default=None)

    args = parser.parse_args()

    return args

def _init_dataset(path, config, reconstruction_model=None, bsize=None):

    if (isinstance(reconstruction_model, list) or isinstance(reconstruction_model, tuple)):
        if (len(reconstruction_model) == 2):
            reconstruction_model_config = None
            reconstruction_model_type, reconstruction_model_weights = reconstruction_model
        else:
            reconstruction_model_config, reconstruction_model_type, reconstruction_model_weights = reconstruction_model
    else:
        reconstruction_model_config = None
        reconstruction_model_weights = None
        reconstruction_model_type = None
    
    if isinstance(config, str):
        config = json.load(open(config, 'r'))
        dset_config = copy.deepcopy(config['dataset'])
        if reconstruction_model_config is None:
            reconstruction_model_config = copy.deepcopy(config.get('reconstruction_model', None))
    else:
        dset_config = copy.deepcopy(config['dataset'])
        if reconstruction_model_config is None:
            reconstruction_model_config = copy.deepcopy(config.get('reconstruction_model', None))

    if bsize is not None:
        dset_config['batch_size'] = bsize
    
    _ = dset_config.pop('return_normalization_parameters', None)
    _ = dset_config.pop('return_case_names', None)

    if (not ('dataset_kwargs' in dset_config)):
        dset_config['dataset_kwargs'] = {}
    dset_config['dataset_kwargs']['return_sensor_values'] = False
    dset_config['dataset_kwargs']['p_gt_substitute'] = 1.0
    
    if (reconstruction_model_config is not None) and (reconstruction_model_type is not None) and (reconstruction_model_weights is not None):
        dm = extract_dataset_metadata(path, config['dataset'].get('full_field_mask',None))
        output_shape = (None, dm['n_gt_vars'], *dm['grid_shape'])
        
        try:
            input_shape = (None, len(list(itertools.chain.from_iterable(dset_config['sensor_masks']))))
        except:
            sensor_locs = dm['sensor_locations']
            n_sensors = sensor_locs[next(iter(sensor_locs))].shape[0]
            n_fieldvars = len(dm['variable_names'])
            input_shape = (None, n_sensors * n_fieldvars)
            
        reconstruction_model = _init_reconstruction_model([[None,reconstruction_model_type,reconstruction_model_config,reconstruction_model_weights]], input_shape, output_shape)[0]
    dset_config['dataset_kwargs']['reconstruction_model'] = reconstruction_model
    if reconstruction_model is not None:
        dset_config['dataset_kwargs']['return_reconstructed_separately'] = True
    
    if hasattr(reconstruction_model, 'output_shape') and len(reconstruction_model.output_shape) == 2:
        model_outputs_grid = False
    else:
        model_outputs_grid = True
        
    train_dataset, test_dataset, _ = TimeMarchingDataset.from_hdf5(path,
                                                        return_normalization_parameters=True,
                                                        return_case_names=True,
                                                        retain_variable_dimension=model_outputs_grid,
                                                        reshape_to_grid=model_outputs_grid,
                                                        **dset_config
                                                        )
    return train_dataset, test_dataset

def _init_models(x, output_shape):

    model_initializers = {
        'fno': lambda **kwargs: FNO(
            output_shape[1:],
            out_channels=output_shape[1],
            return_layer=False,
            **kwargs
        ),
        'unet': lambda **kwargs: UNet(
            nx=output_shape[2],
            ny=output_shape[3],
            in_channels=output_shape[1],
            out_channels=output_shape[1],
            return_layer=False,
            **kwargs
        )
    }

    strip_alphanumeric_pattern = re.compile('[\W_]+')

    models=[]
    for _, model_type, config, weights in x:
        if isinstance(config, str):
            model_config = json.load(open(config,'r'))['model']
        else:
            model_config = config
        model = model_initializers[strip_alphanumeric_pattern.sub('',model_type.lower())](**model_config)
        model.load_weights(weights)
        models.append(model)

    return models

def predict(models,dataset,identifiers=None):
    predictions = []

    print('Running eval...')
    if identifiers is None:
        identifier = list(range(len(models)))

    n_batches = len(dataset)

    inputs_gt = []
    inputs_rc = []
    norm_params_inp = []
    norm_params_target = []
    case_names = []
    targets_gt = []
    targets_rc = []
    pbar = tqdm(total = n_batches)
    pbar.set_description('Copying')
    for x,y,r,n,c in iter(test_dataset):
        inputs_gt.append(x)
        inputs_rc.append(r[:,0])
        targets_gt.append(y)
        targets_rc.append(r[:,1])
        norm_params_inp.append(n[:,0])
        norm_params_target.append(n[:,1])
        case_names.append(c[:,0])
        pbar.update(1)
    results = {
        'targets_gt': np.concatenate(targets_gt,0),
        'targets_rc': np.concatenate(targets_rc,0),
        'case_names': np.concatenate(case_names,0),
        'norm_params_inp': np.concatenate(norm_params_inp,0),
        'norm_params_target': np.concatenate(norm_params_target,0),
        'inputs_gt': np.concatenate(inputs_gt,0),
        'inputs_rc': np.concatenate(inputs_rc,0),
        'predictions_gt': {},
        'predictions_rc': {}
    }
    pbar.close()
    
    for model, identifier in zip(models, identifiers):
        pbar = tqdm(total = n_batches)
        pbar.set_description(str(identifier))
        preds_gt = []
        preds_rc = []
        for x,_,r,_,_ in iter(test_dataset):
            preds_gt.append(model(x).numpy())
            preds_rc.append(model(r[:,0]).numpy())
            pbar.update(1)
            
        results['predictions_gt'][identifier] = np.concatenate(preds_gt,0)
        results['predictions_rc'][identifier] = np.concatenate(preds_rc,0)
        pbar.close()
        
    return results

def pcterror(yp,yt,eps=1e-7):
    return 100*(yp-yt)/(1e-7 + yt)

def _plot_field(x,y,f,cmap,crange,obj_vertices,extents,fname):
    xmin, xmax, ymin, ymax = extents
    plt.figure()
    plt.tripcolor(x,y,f,shading='gouraud',cmap=cmap,vmin=crange[0],vmax=crange[1])
    plt.fill(obj_vertices[0], obj_vertices[1], color='purple')
    plt.axis('equal')
    plt.axis('off')
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    return
        
def plot(snapshot_ids, results, dataset_metadata, domain_extents, save_folder = None, prefix = None, colorbar_limits = None, proportional_colobar_limits = True, errorplot_colorbar_limits = None, percentage_error_threshold=np.inf, max_magnitude_threshold=0.0):
    if save_folder is None:
        save_folder = './'
    if prefix is None:
        prefix = ''
    if colorbar_limits is None:
        colorbar_limits = [0.075 for _ in snapshot_ids]
    elif len(colorbar_limits) == 1:
        colorbar_limits = [colorbar_limits[0] for _ in snapshot_ids]
    if errorplot_colorbar_limits is None:
        errorplot_colorbar_limits = [0,100]

    xmin, xmax, ymin, ymax = domain_extents

    maefn = lambda yt, yp: float(tf.reduce_mean(tf.abs(yp - yt)))
        
    file_ext = '.pdf'
    stats = {}
    for idx, climit in zip(snapshot_ids, colorbar_limits):
        normp_inp = tf.gather(results['norm_params_inp'][idx], dataset_metadata['target_var_indices']).numpy()
        normp_tar = tf.gather(results['norm_params_target'][idx], dataset_metadata['target_var_indices']).numpy()
        inp_gt = np.reshape(results['inputs_gt'][idx], [-1,dataset_metadata['n_gt_vars']]) + normp_inp
        inp_rc = np.reshape(results['inputs_rc'][idx], [-1,dataset_metadata['n_gt_vars']]) + normp_inp
        target = np.reshape(results['targets_gt'][idx],[-1,dataset_metadata['n_gt_vars']]) + normp_tar
        case_name = results['case_names'][idx]#.decode('utf-8')

        vmax = climit*np.max(np.abs(target)) if proportional_colobar_limits else climit
        vmin = -vmax
        stats[idx] = {'cbar_limits': [vmin, vmax], 'metrics':{}}
        x = np.real(dataset_metadata['full_field_locations'][case_name])
        y = np.imag(dataset_metadata['full_field_locations'][case_name])

        xmask = np.logical_and(x>xmin, x<xmax)
        ymask = np.logical_and(y<ymax, y>ymin)
        spatialmask = np.logical_and(xmask, ymask)
        spatialmasked_indices = np.where(spatialmask)
        x = x[spatialmasked_indices]
        y = y[spatialmasked_indices]

        inp_gt = inp_gt[spatialmasked_indices]
        inp_rc = inp_rc[spatialmasked_indices]
        target = target[spatialmasked_indices]

        sensor_coords_x = np.real(dataset_metadata['sensor_locations'][case_name])
        sensor_coords_y = np.imag(dataset_metadata['sensor_locations'][case_name])

        obj_vertices = [sensor_coords_x[dataset_metadata['n_vsensors']:],sensor_coords_y[dataset_metadata['n_vsensors']:]]
        
        for identifier in results['predictions_gt']:
            pred_gt = np.reshape(results['predictions_gt'][identifier][idx],[-1,dataset_metadata['n_gt_vars']]) + normp_tar
            pred_gt = pred_gt[spatialmasked_indices]

            pred_rc = np.reshape(results['predictions_rc'][identifier][idx],[-1,dataset_metadata['n_gt_vars']]) + normp_tar
            pred_rc = pred_rc[spatialmasked_indices]

            plot_tensors = zip(
                np.transpose(pred_gt, [1,0]),
                np.transpose(pred_rc, [1,0]),
                np.transpose(pcterror(pred_gt, target), [1,0]),
                np.transpose(pcterror(pred_rc, target), [1,0])
            )

            plot_tensor_names = (
                'predgt',
                'predrc',
                'pcterror-predgt-target',
                'pcterror-predrc-target'
            )

            cmaps = ['rainbow' for _ in range(2)] + ['Blues' for _ in range(2)]
            cranges = [[vmin,vmax] for _ in range(2)] + [errorplot_colorbar_limits for _ in range(2)]

            stats[idx]['metrics'][identifier] = {}
            for k, tensors in enumerate(plot_tensors):
                varname = dataset_metadata['variable_names'][dataset_metadata['target_var_indices'][k]]

                for i,(tensor, tname, cmap, crange) in enumerate(zip(tensors, plot_tensor_names, cmaps, cranges)):
                    savepath = save_folder + '/' + '_'.join([prefix,f's{idx}',str(identifier),varname,tname])
                    _plot_field(x,y,tensor,cmap,crange,obj_vertices, domain_extents, savepath + file_ext)
                    
                stats[idx]['metrics'][identifier][varname] = {
                    'mape-predgt-target': mape_with_threshold(tensors[0], target[:,k], percentage_error_threshold, max_magnitude_threshold),
                    'mape-predrc-target': mape_with_threshold(tensors[1], target[:,k], percentage_error_threshold, max_magnitude_threshold),
                    'mae-predgt-target': maefn(tensors[0], target[:,k]),
                    'mae-predrc-target': maefn(tensors[1], target[:,k])
                }

        plot_tensors = zip(
            np.transpose(inp_gt, [1,0]),
            np.transpose(inp_rc, [1,0]),
            np.transpose(target, [1,0]),
            np.transpose(pcterror(inp_rc, inp_gt), [1,0]),
        )

        plot_tensor_names = (
            'inputgt',
            'inputrc',
            'target',
            'pcterror-inprc-inpgt'
        )

        cmaps = ['rainbow' for _ in range(3)] + ['Blues' for _ in range(1)]
        cranges = [[vmin,vmax] for _ in range(3)] + [errorplot_colorbar_limits for _ in range(1)]
        stats[idx]['mape-inprc-inpgt'] = {}
        stats[idx]['mae-inprc-inpgt'] = {}
        for k, tensors in enumerate(plot_tensors):
            varname = dataset_metadata['variable_names'][dataset_metadata['target_var_indices'][k]]
            
            for i, (tensor, tname, cmap, crange) in enumerate(zip(tensors, plot_tensor_names, cmaps, cranges)):
                savepath = save_folder + '/' + '_'.join([prefix,f's{idx}',varname,tname])
                _plot_field(x,y,tensor,cmap,crange,obj_vertices, domain_extents, savepath + file_ext)

            stats[idx]['mape-inprc-inpgt'][varname] = mape_with_threshold(tensors[1], tensors[0], percentage_error_threshold)
            stats[idx]['mae-inprc-inpgt'][varname] = maefn(tensors[1], tensors[0])

    return stats
    
def compute_error(results, dataset_metadata, domain_extents, save_folder=None, prefix=None, percentage_error_threshold=np.inf, max_magnitude_threshold=0.0):
    if prefix is None:
        prefix = ''

    if save_folder is None:
        save_folder='./'
        
    xmin, xmax, ymin, ymax = domain_extents

    all_indices = []
    case_names = results['case_names']
    sensor_locations = dataset_metadata['sensor_locations']
    full_field_locations = dataset_metadata['full_field_locations']
    n_vsensors = dataset_metadata['n_vsensors']
    print('Error calculation...')
    for k, cname in enumerate(tqdm(case_names)):
        _cname = cname#.decode('utf-8')
        sensor_coords_x = np.real(sensor_locations[_cname])[n_vsensors:]
        sensor_coords_y = np.imag(sensor_locations[_cname])[n_vsensors:]
        sensor_coords = np.stack([sensor_coords_x, sensor_coords_y],-1)
        x = np.real(full_field_locations[_cname])
        y = np.imag(full_field_locations[_cname])
        
        xmask = np.logical_and(x>xmin, x<xmax)
        ymask = np.logical_and(y<ymax, y>ymin)
        spatialmask = np.logical_and(xmask, ymask)
        spatialmask = np.logical_and(spatialmask,
                np.logical_not(check_in_polygon(np.stack([x,y],-1), sensor_coords)))

        spatialmasked_indices = np.where(spatialmask)[0]
        indices = np.stack([
            k*np.ones(spatialmasked_indices.shape[0]).astype(spatialmasked_indices.dtype),
            spatialmasked_indices,
            np.zeros(spatialmasked_indices.shape[0]).astype(spatialmasked_indices.dtype)
        ], -1)
        all_indices.append(indices)

    all_indices = np.concatenate(all_indices,0)

    reshape_normp = lambda n, p: np.reshape(n, list(n.shape) + [1 for _ in range(len(p.shape) - len(n.shape))])
    normp_inp = reshape_normp(
        results['norm_params_inp'][:,dataset_metadata['target_var_indices']],
        results['targets_gt']
    )
    normp_tar = results['norm_params_target'][:,dataset_metadata['target_var_indices']]
    
    preds_gt_masked = {}
    preds_rc_masked = {}
    for k in results['predictions_gt']:
        reshaped_normp_tar = reshape_normp(
            normp_tar,
            results['predictions_gt'][k]
        )
        pred_gt = results['predictions_gt'][k] + reshaped_normp_tar
        masked_pred_gt = gather_masked_indices(pred_gt, all_indices, n_gt_vars=dataset_metadata['n_gt_vars'])
        
        pred_rc = results['predictions_rc'][k] + reshaped_normp_tar
        masked_pred_rc = gather_masked_indices(pred_rc, all_indices, n_gt_vars=dataset_metadata['n_gt_vars'])
        
        preds_gt_masked[k] = masked_pred_gt
        preds_rc_masked[k] = masked_pred_rc
        
    targets_masked = gather_masked_indices(
        results['targets_gt'] + normp_inp,
        all_indices,
        n_gt_vars=dataset_metadata['n_gt_vars']
    )
    inputs_gt_masked = gather_masked_indices(
        results['inputs_gt'] + normp_inp,
        all_indices,
        n_gt_vars=dataset_metadata['n_gt_vars']
    )
    inputs_rc_masked = gather_masked_indices(
        results['inputs_rc'] + normp_inp,
        all_indices,
        n_gt_vars=dataset_metadata['n_gt_vars']
    )

    maefn = lambda yt, yp: float(tf.reduce_mean(tf.abs(yp - yt)))
    maes_preds_gt_targets = {k:maefn(preds_gt_masked[k], targets_masked) for k in preds_gt_masked}
    mapes_preds_gt_targets = {k:mape_with_threshold(preds_gt_masked[k], targets_masked, percentage_error_threshold, max_magnitude_threshold) for k in preds_gt_masked}
    maes_preds_rc_targets = {k:maefn(preds_rc_masked[k],targets_masked) for k in preds_rc_masked}
    mapes_preds_rc_targets = {k:mape_with_threshold(preds_rc_masked[k], targets_masked, percentage_error_threshold, max_magnitude_threshold) for k in preds_rc_masked}
    maes_inpgt_inprc = maefn(inputs_gt_masked, inputs_rc_masked)
    mapes_inpgt_inprc = mape_with_threshold(inputs_rc_masked, inputs_gt_masked, percentage_error_threshold, max_magnitude_threshold)
    print(f'MAE preds_gt - targets: {maes_preds_gt_targets}')
    print(f'MAE preds_rc - targets: {maes_preds_rc_targets}')
    print(f'MAPE preds_gt - targets: {mapes_preds_gt_targets}')
    print(f'MAPE preds_rc - targets: {mapes_preds_rc_targets}')
    
    summary = {k:{
        'mae-predgt-target':maes_preds_gt_targets[k],
        'mape-predgt-target':mapes_preds_gt_targets[k],
        'mae-predrc-target':maes_preds_rc_targets[k],
        'mape-predrc-target':mapes_preds_rc_targets[k]
    } for k in maes_preds_gt_targets}
    summary['mae-inpgt-inprc'] = maes_inpgt_inprc
    summary['mape-inpgt-inprc'] = mapes_inpgt_inprc
    return summary

if __name__ == '__main__':
    mpl.rcParams['figure.dpi'] = 900
    tf.keras.backend.set_image_data_format('channels_first')
    args = _parse_args_spatiotemporal()

    ffm = json.load(open(args.model[0][2],'r'))['dataset'].get('full_field_mask',None)
    dataset_metadata = extract_dataset_metadata(args.dataset,ffm)
    
    train_dataset, test_dataset = _init_dataset(args.dataset, args.model[0][2], args.reconstruction_model[1:], args.bsize)
    
    output_shape = (None,dataset_metadata['n_gt_vars'], *dataset_metadata['grid_shape'])
    models = _init_models(args.model, output_shape)

    results = predict(models, train_dataset if args.traindata else test_dataset, [x[0] for x in args.model])
    
    plt_stats = plot(args.s, results, dataset_metadata, args.e, args.o, args.prefix, args.c, args.l, args.p, args.pcterrt, args.magnitudet)
    
    error_summary = compute_error(results, dataset_metadata, args.e, args.o, args.prefix, args.pcterrt, args.magnitudet)
    
    fname = args.o + '/' + '_'.join([args.prefix, 'errormetrics'])+ '.json'
    for identifier in [x[0] for x in args.model]:
        error_summary[identifier]['snapshots'] = {}
    error_summary['colorbar_limits'] = {}
    error_summary['snapshot_global_metrics'] = {}
    for idx in plt_stats:
        snapshot_global_metrics = {}
        excl_list = ['metrics', 'cbar_limits']
        error_summary['snapshot_global_metrics'][idx] = {}
        for k in plt_stats[idx]:
            if (not (k in excl_list)):
                error_summary['snapshot_global_metrics'][idx][k] = plt_stats[idx][k]
        for identifier in plt_stats[idx]['metrics']:
            varnames_list = list(plt_stats[idx]['metrics'][identifier].keys())
            metrics_list = list(plt_stats[idx]['metrics'][identifier][varnames_list[0]].keys())
            error_summary[identifier]['snapshots'][idx] = {varname:{metric:plt_stats[idx]['metrics'][identifier][varname][metric] for metric in metrics_list} for varname in varnames_list}
        error_summary['colorbar_limits'][idx] = plt_stats[idx]['cbar_limits']
    
    json.dump(error_summary, open(fname,'w'), indent=4)
