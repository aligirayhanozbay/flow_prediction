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

def _parse_args():

    parser=argparse.ArgumentParser('Generate plots for the spatial-only case')
    
    parser.add_argument('dataset', type=str, help='path to the dataset file')
    parser.add_argument('-o', type=str, default='./', help='output directory')
    parser.add_argument('-s', nargs='*', type=int, help='indices of snapshots to use', default=[0])
    parser.add_argument('-m','--model',
                        action='append',
                        nargs=4,
                        metavar=('identifier', 'model_type', 'config', 'weights'))
    parser.add_argument('--prefix', type=str, default=None, help='prefix for all output files')
    parser.add_argument('--traindata', action='store_true', help='use training data instead of test data')

    args = parser.parse_args()

    return args

def _init_dataset(path, config):
    if isinstance(config, str):
        dset_config = json.load(open(config, 'r'))['dataset']
    else:
        dset_config = copy.deepcopy(config)

    
    _ = dset_config.pop('return_normalization_parameters', None)
    _ = dset_config.pop('return_case_names', None)
    train_dataset, test_dataset = ShallowDecoderDataset(path,
                                                        return_normalization_parameters=True,
                                                        return_case_names=True,
                                                        **dset_config
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
        identifier = list(range(len(models)))

    n_batches = int(dataset.cardinality())

    inputs = []
    norm_params = []
    case_names = []
    ground_truths = []
    pbar = tqdm(total = n_batches)
    pbar.set_description('Copying')
    for x,y,n,c in iter(test_dataset):
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
        for x,_,_,_ in iter(test_dataset):
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

    md['n_vsensors'] = {75:25, 34:9, 16:4}[md['sensor_locations'][list(md['sensor_locations'].keys())[0]].shape[-1]]

    return md
        
def plot(snapshot_ids, results, dataset_metadata, save_folder = None, prefix = None):
    if save_folder is None:
        save_folder = './'
    if prefix is None:
        prefix = ''
    
    xmin = -1.5
    xmax = 6.0
    ymin = -1.5
    ymax = 1.5
    file_ext = '.png'
    for idx in snapshot_ids:
        normp = tf.gather(results['norm_params'][idx], dataset_metadata['target_var_indices'])
        gt = np.reshape(results['ground_truths'][idx],[-1,dataset_metadata['n_gt_vars']]) + normp
        case_name = results['case_names'][idx].decode('utf-8')

        vmax = 0.075*np.max(np.abs(gt))
        vmin = -vmax
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
            pred = np.reshape(results['predictions'][identifier][idx],[-1,dataset_metadata['n_gt_vars']]) + normp
            pred = pred.numpy()[spatialmasked_indices]

            for k,(yt,yp) in enumerate(zip(tf.transpose(gt,(1,0)), tf.transpose(pred,(1,0)))):
                varname = dataset_metadata['variable_names'][dataset_metadata['target_var_indices'][k]]
                plt.figure()
                plt.tripcolor(x,y,yp,shading='gouraud', cmap='rainbow', vmin = vmin, vmax = vmax)
                plt.fill(sensor_coords_x[dataset_metadata['n_vsensors']:], sensor_coords_y[dataset_metadata['n_vsensors']:], color='purple')
                plt.axis('equal')
                plt.axis('off')
                plt.xlim([xmin,xmax])
                plt.ylim([ymin,ymax])
                savepath = save_folder + '/' + '_'.join([prefix,f's{idx}',str(identifier),varname])
                plt.savefig(savepath + file_ext, bbox_inches='tight')
                plt.close()

                plt.figure()
                plt.tripcolor(x,y,100*(yp-yt)/(1e-7 + yt),shading='gouraud', cmap='RdBu', vmin = -100, vmax = 100)
                plt.fill(sensor_coords_x[dataset_metadata['n_vsensors']:], sensor_coords_y[dataset_metadata['n_vsensors']:], color='purple')
                plt.axis('equal')
                plt.axis('off')
                plt.xlim([xmin,xmax])
                plt.ylim([ymin,ymax])
                savepath = save_folder + '/' + '_'.join([prefix,f's{idx}',str(identifier),varname+'pcterror'])
                plt.savefig(savepath + file_ext, bbox_inches='tight')
                plt.close()

        for k, yt in enumerate(tf.transpose(gt, (1,0))):
            plt.figure()
            plt.tripcolor(x,y,yt,shading='gouraud', cmap='rainbow', vmin = vmin, vmax = vmax)
            plt.fill(sensor_coords_x[dataset_metadata['n_vsensors']:], sensor_coords_y[dataset_metadata['n_vsensors']:], color='purple')
            plt.axis('equal')
            plt.axis('off')
            plt.colorbar()
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            savepath = save_folder + '/' + '_'.join([prefix,f's{idx}','gt',varname])
            plt.savefig(savepath + file_ext, bbox_inches='tight')
            plt.close()

            
            

if __name__ == '__main__':
    mpl.rcParams['figure.dpi'] = 900
    tf.keras.backend.set_image_data_format('channels_first')
    args = _parse_args()

    train_dataset, test_dataset = _init_dataset(args.dataset, args.model[0][2])

    ffm = json.load(open(args.model[0][2],'r'))['dataset'].get('full_field_mask',None)
    dataset_metadata = extract_dataset_metadata(args.dataset,ffm)

    output_shape = (None,dataset_metadata['n_gt_vars'], *dataset_metadata['grid_shape'])
    models = _init_models(args.model, test_dataset.element_spec[0].shape, output_shape)

    results = predict(models, train_dataset if args.traindata else test_dataset, [x[0] for x in args.model])

    plot(args.s, results, dataset_metadata, args.o, args.prefix)
    
    

    
