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
from tqdm import tqdm

from flow_prediction.models.SD_UNet import SD_UNet
from flow_prediction.models.ShallowDecoder import original_shallow_decoder
from flow_prediction.models.SD_FNO import SD_FNO
from flow_prediction.data.training.ShallowDecoderDataset import ShallowDecoderDataset, _extract_variable_names, _get_unique_variable_mask_ids

def _parse_args_spatial():

    parser=argparse.ArgumentParser('Generate plots for the spatial-only case')
    
    parser.add_argument('dataset', type=str, help='path to the dataset file')
    parser.add_argument('-o', type=str, default='./', help='output directory')
    parser.add_argument('-s', nargs='*', type=int, help='indices of snapshots to use', default=[0])
    parser.add_argument('-c', nargs='*', type=float, help='provide a value for each snapshot, or a single value for all snapshots. clips the colorbars in the plots. e.g. supplying 0.075 for a snapshot means that vmax=-vmin=0.075*max(abs(values in the snapshot)). default is 0.075.', default=[0.075])
    parser.add_argument('-l', action='store_false', help='interpret colorbar extents as literal values instead of a % of max(abs(values in the snapshot))')
    parser.add_argument('-p', nargs=2, type=float, help='colobar limits for the % error plot', default=[0,100])
    parser.add_argument('-m','--model',
                        action='append',
                        nargs=4,
                        metavar=('identifier', 'model_type', 'config', 'weights'))
    parser.add_argument('-e', nargs=4, type=float, help='extents of the plotting domain',
                        metavar=('xmin', 'xmax', 'ymin', 'ymax'),
                        default=[-1.5,6.0,-1.5,1.5])
    parser.add_argument('--cmap', default='rainbow', type=str, help='color map to use. default is rainbow. see matplotlib documentation.')
    parser.add_argument('--prefix', type=str, default=None, help='prefix for all output files')
    parser.add_argument('--traindata', action='store_true', help='use training data instead of test data')
    parser.add_argument('--pcterrt', type=float, help='% errors above this value will be discarded in metrics calculations. can be used to filter out extremely high % errors caused due very small denominators.', default=np.inf)
    parser.add_argument('--magnitudet', type=float, help='float between 0 and 1. if set to e.g. 0.05, for some field f, if a ground truth data pt has absolute value equal to or smaller than 5% of max(|f|), that point is excluded from error metrics', default=0.0)
    parser.add_argument('--bsize', type=int, help='override the batch size used by the dataset', default=None)

    args = parser.parse_args()

    return args

def _init_dataset(path, config, bsize=None, **other_args):
    if isinstance(config, str):
        dset_config = json.load(open(config, 'r'))['dataset']
    else:
        dset_config = copy.deepcopy(config)

    if bsize is not None:
        dset_config['batch_size'] = bsize
    
    _ = dset_config.pop('return_normalization_parameters', None)
    _ = dset_config.pop('return_case_names', None)
    train_dataset, test_dataset = ShallowDecoderDataset(path,
                                                        return_normalization_parameters=True,
                                                        return_case_names=True,
                                                        **dset_config,
                                                        **other_args
                                                        )
    return train_dataset, test_dataset

def _init_models(x, input_shape, output_shape):

    model_initializers = {
        'sdfno': lambda **kwargs: SD_FNO(
            input_shape[-1],
            output_shape[2:],
            out_channels=output_shape[1],
            **kwargs
        ),
        'sdunet': lambda **kwargs: SD_UNet(
            input_shape[-1],
            output_shape[2:],
            out_channels=output_shape[1],
            **kwargs
        ),
        'sd': lambda **kwargs: original_shallow_decoder(
            input_shape[-1],
            int(np.prod(output_shape[1:])),
            learning_rate=1e-3,
            **kwargs
        )
    }
    model_initializers['shallowdecoder'] = model_initializers['sd']

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
        identifiers = list(range(len(models)))

    n_batches = int(dataset.cardinality())

    inputs = []
    norm_params = []
    case_names = []
    ground_truths = []
    pbar = tqdm(total = n_batches)
    pbar.set_description('Copying')
    for x,y,n,c in iter(dataset):
        inputs.append(x.numpy())
        norm_params.append(n.numpy())
        case_names.append(c.numpy())
        ground_truths.append(y.numpy())
        pbar.update(1)
    results = {
        'ground_truths': np.concatenate(ground_truths,0),
        'case_names': np.concatenate(case_names,0),
        'norm_params': np.concatenate(norm_params,0),
        'inputs': np.concatenate(inputs,0),
        'predictions': {}
    }
    pbar.close()
    
    for model, identifier in zip(models, identifiers):
        pbar = tqdm(total = n_batches)
        pbar.set_description(str(identifier))
        preds = []
        for x,_,_,_ in iter(dataset):
            preds.append(model(x).numpy())
            pbar.update(1)
            
        results['predictions'][identifier] = np.concatenate(preds,0)
        pbar.close()
        
    return results

def extract_dataset_metadata(dataset_path, full_field_mask = None):
    md = {}
    with h5py.File(dataset_path, 'r') as f:
        md['variable_names'] = _extract_variable_names(f)
        md['variable_name_idx_map'] = {varname:k for k, varname in enumerate(md['variable_names'])}
        md['sensor_locations'] = {case:np.array(f[case]['sensor_locations_z']) for case in f}
        md['full_field_locations'] = {case:np.array(f[case]['full_field_locations_z']) for case in f}
        md['grid_shape'] = f.attrs['grid_shape']
    
    if full_field_mask is not None:
        md['n_gt_vars'] = len(full_field_mask)
        md['target_var_indices'] = sorted([md['variable_name_idx_map'][varname] for varname in full_field_mask])
    else:
        md['n_gt_vars'] = len(md['variable_names'])
        md['target_var_indices'] = list(range(len(md['variable_names'])))

    md['n_vsensors'] = {75:25, 34:9, 16:4}[md['sensor_locations'][list(md['sensor_locations'].keys())[0]].shape[-1]]#TODO: make this more general somehow

    return md
        
def plot(snapshot_ids, results, dataset_metadata, domain_extents, save_folder = None, prefix = None, colorbar_limits = None, proportional_colobar_limits = True, errorplot_colorbar_limits = None, percentage_error_threshold=np.inf, max_magnitude_threshold=0.0, cmap = None):
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

    cmap = cmap or 'rainbow'
    xmin, xmax, ymin, ymax = domain_extents
        
    file_ext = '.pdf'
    stats = {}
    for idx, climit in zip(snapshot_ids, colorbar_limits):
        normp = tf.gather(results['norm_params'][idx], dataset_metadata['target_var_indices'])
        gt = np.reshape(results['ground_truths'][idx],[-1,dataset_metadata['n_gt_vars']]) + normp
        case_name = results['case_names'][idx].decode('utf-8')

        vmax = climit*np.max(np.abs(gt)) if proportional_colobar_limits else climit
        vmin = -vmax
        stats[idx] = {'cbar_limits': [vmin, vmax], 'metrics':{}}
        x = np.real(dataset_metadata['full_field_locations'][case_name])
        #xmin, xmax = np.min(x), np.max(x)
        y = np.imag(dataset_metadata['full_field_locations'][case_name])
        #ymin, ymax = np.min(y), np.max(y)

        xmask = np.logical_and(x>xmin, x<xmax)
        ymask = np.logical_and(y<ymax, y>ymin)
        spatialmask = np.logical_and(xmask, ymask)
        spatialmasked_indices = np.where(spatialmask)
        x = x[spatialmasked_indices]
        y = y[spatialmasked_indices]
        gt = gt.numpy()[spatialmasked_indices]

        sensor_coords_x = np.real(dataset_metadata['sensor_locations'][case_name])
        sensor_coords_y = np.imag(dataset_metadata['sensor_locations'][case_name])
        
        for identifier in results['predictions']:
            pred = results['predictions'][identifier][idx]
            if len(pred.shape) > 1:
                _newaxes = list(range(1,len(pred.shape))) + [0]
                pred = pred.transpose(_newaxes)
            pred = np.reshape(pred,[-1,dataset_metadata['n_gt_vars']]) + normp
            pred = pred.numpy()[spatialmasked_indices]

            stats[idx]['metrics'][identifier] = {}
            for k,(yt,yp) in enumerate(zip(tf.transpose(gt,(1,0)), tf.transpose(pred,(1,0)))):
                varname = dataset_metadata['variable_names'][dataset_metadata['target_var_indices'][k]]
                plt.figure()
                plt.tripcolor(x,y,yp,shading='gouraud', cmap=cmap, vmin = vmin, vmax = vmax)
                plt.fill(sensor_coords_x[dataset_metadata['n_vsensors']:], sensor_coords_y[dataset_metadata['n_vsensors']:], color='purple')
                plt.axis('equal')
                plt.axis('off')
                plt.xlim([xmin,xmax])
                plt.ylim([ymin,ymax])
                savepath = save_folder + '/' + '_'.join([prefix,f's{idx}',str(identifier),varname])
                plt.savefig(savepath + file_ext, bbox_inches='tight')
                plt.close()

                plt.figure()
                plt.tripcolor(x,y,100*np.abs((yp-yt)/(1e-7 + yt)),shading='gouraud', cmap='Blues', vmin = 0, vmax = 100)
                plt.fill(sensor_coords_x[dataset_metadata['n_vsensors']:], sensor_coords_y[dataset_metadata['n_vsensors']:], color='purple')
                plt.axis('equal')
                plt.axis('off')
                plt.xlim([xmin,xmax])
                plt.ylim([ymin,ymax])
                savepath = save_folder + '/' + '_'.join([prefix,f's{idx}',str(identifier),varname+'pcterror'])
                plt.savefig(savepath + file_ext, bbox_inches='tight')
                plt.close()

                stats[idx]['metrics'][identifier][varname]= {
                    'mapes': mape_with_threshold(yp, yt, percentage_error_threshold, max_magnitude_threshold),
                    'maes': float(tf.reduce_mean(tf.abs(yp - yt)))
                    }
                

        for k, yt in enumerate(tf.transpose(gt, (1,0))):
            varname = dataset_metadata['variable_names'][dataset_metadata['target_var_indices'][k]]
            plt.figure()
            plt.tripcolor(x,y,yt,shading='gouraud', cmap=cmap, vmin = vmin, vmax = vmax)
            plt.fill(sensor_coords_x[dataset_metadata['n_vsensors']:], sensor_coords_y[dataset_metadata['n_vsensors']:], color='purple')
            plt.axis('equal')
            plt.axis('off')
            plt.colorbar()
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            savepath = save_folder + '/' + '_'.join([prefix,f's{idx}','gt',varname])
            plt.savefig(savepath + file_ext, bbox_inches='tight')
            plt.close()

    return stats

def gather_masked_indices(x, sm, n_gt_vars = 1, norm_params = 0.0):
    xshape = tf.shape(x)
    bsize = xshape[0]

    if isinstance(norm_params, tf.Tensor) or isinstance(norm_params, np.ndarray) and (len(norm_params.shape) != 3):
        norm_params = tf.reshape(norm_params, [bsize, 1, n_gt_vars])
    
    if tf.rank(x) == 4:
        x = tf.transpose(x, [0,2,3,1])
    
    x = tf.reshape(x,[bsize,-1,n_gt_vars])
    x = tf.gather_nd(x + norm_params, sm)
    return x

def check_in_polygon(pts, vertices):
    path = mplPath.Path(vertices)
    return path.contains_points(pts)

def mape_with_threshold(yp, yt, pcterror_threshold=np.inf, max_magnitude_threshold=0.0, eps=1e-7):
    pct_errors = 100*tf.abs((yp-yt)/(eps + yt))
    pcterror_mask = pct_errors < pcterror_threshold
    max_magnitude_mask = tf.logical_not(tf.abs(yt) < (max_magnitude_threshold*tf.reduce_max(tf.abs(yt))))
    filtering_indices = tf.where(tf.logical_and(pcterror_mask, max_magnitude_mask))
    filtered_pcterrors = tf.gather_nd(pct_errors, filtering_indices)
    return float(tf.reduce_mean(filtered_pcterrors))
    
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
        _cname = cname.decode('utf-8')
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
            spatialmasked_indices
        ], -1)
        all_indices.append(indices)

    all_indices = np.concatenate(all_indices,0)
    
    normp = results['norm_params'][:,dataset_metadata['target_var_indices']]
    reshape_normp = lambda n, p: np.reshape(n, list(n.shape) + [1 for _ in range(len(p.shape) - len(n.shape))])
    
    preds_masked = {}
    for k in results['predictions']:
        pred = results['predictions'][k]
        reshaped_normp = reshape_normp(normp, pred)
        masked_pred = gather_masked_indices(pred + reshaped_normp, all_indices, n_gt_vars=dataset_metadata['n_gt_vars'])
        preds_masked[k] = masked_pred
    
    gts_masked = gather_masked_indices(
        results['ground_truths'],
        all_indices,
        n_gt_vars=dataset_metadata['n_gt_vars'],
        norm_params=reshape_normp(normp, results['ground_truths'])
    )

    
    maes = {}
    mapes = {}
    varnames = [dataset_metadata['variable_names'][vidx] for vidx in dataset_metadata['target_var_indices']]
    for k in preds_masked:
        maes[k] = {}
        mapes[k] = {}
        for vidx in range(gts_masked.shape[-1]):
            maes[k][varnames[vidx]] = float(tf.reduce_mean(tf.abs(preds_masked[k][...,vidx] - gts_masked[...,vidx])))
            mapes[k][varnames[vidx]] = mape_with_threshold(preds_masked[k][...,vidx], gts_masked[...,vidx], percentage_error_threshold, max_magnitude_threshold)
        
    
    print(f'MAE: {maes}')
    print(f'MAPE: {mapes}')
    
    summary = {k:{'mae':maes[k], 'mape':mapes[k]} for k in maes}
    return summary



if __name__ == '__main__':
    mpl.rcParams['figure.dpi'] = 900
    tf.keras.backend.set_image_data_format('channels_first')
    args = _parse_args_spatial()
    
    train_dataset, test_dataset = _init_dataset(args.dataset, args.model[0][2], args.bsize)

    ffm = json.load(open(args.model[0][2],'r'))['dataset'].get('full_field_mask',None)
    dataset_metadata = extract_dataset_metadata(args.dataset,ffm)
    
    output_shape = (None,dataset_metadata['n_gt_vars'], *dataset_metadata['grid_shape'])
    models = _init_models(args.model, test_dataset.element_spec[0].shape, output_shape)

    results = predict(models, train_dataset if args.traindata else test_dataset, [x[0] for x in args.model])

    plt_stats = plot(args.s, results, dataset_metadata, args.e, args.o, args.prefix, args.c, args.l, args.p, args.pcterrt, args.magnitudet, cmap = args.cmap)
    error_summary = compute_error(results, dataset_metadata, args.e, args.o, args.prefix, args.pcterrt, args.magnitudet)

    fname = args.o + '/' + '_'.join([args.prefix, 'errormetrics'])+ '.json'
    

    for identifier in error_summary:
        error_summary[identifier]['snapshots'] = {}
    error_summary['colorbar_limits'] = {}
    for idx in plt_stats:
        for identifier in plt_stats[idx]['metrics']:
            varnames_list = list(plt_stats[idx]['metrics'][identifier].keys())
            metrics_list = list(plt_stats[idx]['metrics'][identifier][varnames_list[0]].keys())
            error_summary[identifier]['snapshots'][idx] = {varname:{metric:plt_stats[idx]['metrics'][identifier][varname][metric] for metric in metrics_list} for varname in varnames_list}
        error_summary['colorbar_limits'][idx] = plt_stats[idx]['cbar_limits']
    
    json.dump(error_summary, open(fname,'w'), indent=4)
    
    

    
