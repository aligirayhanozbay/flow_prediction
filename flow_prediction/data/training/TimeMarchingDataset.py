import h5py
import numpy as np
import tensorflow as tf
import itertools
import random
import warnings
from tqdm import tqdm

from .ShallowDecoderDataset import _extract_values
from .SpatiotemporalDataset import split_arrays, _pick_init_arrays

class TimeMarchingDataset(tf.keras.utils.Sequence):
    _train_test_split_method_axes = {'case':0, None:0,'time':1}
    _numeric_dtypes = [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64]
    _float_dtypes = [np.float16, np.float32, np.float64]

    def __init__(self, batch_size, full_field_values, sensor_values = None, normalization_params = None, case_names = None, case_ids = None, temporal_stride = 1, reconstruction_model = None, p_gt_substitute=0.0, return_sensor_values=False, return_reconstructed_separately=False):
        '''
        -batch_size: int. batch size.
        -full_field_values, sensor_values, normalization_params: np.ndarray of shape [n_cases, n_timesteps, ...].
        -case_names, case_ids: np.ndarray of shape [n_cases, n_timesteps].
        -temporal_stride: int.
        -reconstruction_model: Tuple[np.array, tf.keras.Model, int] or Tuple[np.array, tf.keras.Model]. When this is supplied, the inputs for the dataset are not drawn from full_field values, but are instead generated using the model supplied from the sensor_inputs (first element of the tuple). This can be used to create a dataset where the inputs are snapshots predicted by the model supplied at time t, and the ground truths are the actual snapshots for t+k*dt.
        -p_gt_substitute: When sensor_inputs_and_model is not None, determines the probability of substituting an actual snapshot from full_field_values instead of using the reconstructed snapshot. (Providing 1.0 disables the usage of predicted snapshots entirely)
        -return_sensor_values: bool. If True, sensor values are returned
        -return_reconstructed_separately: If True, reconstructed snapshots are returned separately. To disable returning them also as the targets, set p_gt_substitute = 0.0.
        '''
        
        super().__init__()

        self.batch_size = batch_size
        
        self._register_return_arrays(sensor_values, full_field_values, normalization_params, case_names, case_ids, return_sensor_values)

        self.temporal_stride = temporal_stride
        self._case_temporal_indices = self._generate_per_sample_case_temporal_indices()
        assert len(self._case_temporal_indices) == self.n_samples
        self._indices_per_batch = self._make_batch_indices(self.n_samples, self.batch_size)
        assert len(self._indices_per_batch) == len(self)
        
        self.p_gt_substitute = p_gt_substitute
        self.return_reconstructed_separately = return_reconstructed_separately
        if reconstruction_model is not None:
            if hasattr(reconstruction_model, '__getitem__') and len(reconstruction_model) == 2:
                model, pred_bsize = reconstruction_model
            else:
                model = reconstruction_model
                pred_bsize = self.batch_size
            self._reconstructed_fields = self._sensor_inputs_to_predicted_fields(self.sensor_values, model, pred_bsize)
            if self.return_reconstructed_separately:
                self._return_order.insert(1,'_reconstructed_fields')
                if not (self.p_gt_substitute > 0.0):
                    warnings.warn('return_reconstructed_separately was set to True with p_gt_substitute > 0.0. This may still return target snapshots as reconstructed values - set 0.0 to always get ground truth values as target snapshots!')
        else:
            self._reconstructed_fields = None
            

    def __len__(self):
        return int(np.ceil(self.n_samples/self.batch_size))

    def _register_return_arrays(self, sensor_values, full_field_values, normalization_params, case_names, case_ids, return_sensor_values):
        assert (isinstance(full_field_values, np.ndarray) or isinstance(full_field_values, tf.Tensor))
        self.full_field_values = full_field_values
        return_order = ['full_field_values']
        if sensor_values is not None:
            assert sensor_values.shape[:2] == full_field_values.shape[:2]
            self.sensor_values = sensor_values
            if return_sensor_values:
                return_order.append('sensor_values')
        if normalization_params is not None:
            assert full_field_values.shape[:2] == normalization_params.shape[:2]
            self.normalization_params = normalization_params
            return_order.append('normalization_params')
        if case_names is not None:
            assert full_field_values.shape[:2] == case_names.shape[:2]
            self.case_names = case_names
            return_order.append('case_names')
        if case_ids is not None:
            assert full_field_values.shape[:2] == case_ids.shape[:2]
            self.case_ids = case_ids
            return_order.append('case_ids')

        self._return_order = return_order

    @property
    def n_cases(self):
        return self.full_field_values.shape[0]

    @property
    def n_timesteps(self):
        return self.full_field_values.shape[1]

    @property
    def n_samples(self):
        return self.n_cases * (self.n_timesteps-self.temporal_stride)

    def _generate_per_sample_case_temporal_indices(self):
        case_idxs = np.arange(self.n_cases)
        t0 = np.arange(self.n_timesteps-self.temporal_stride)
        t1 = t0 + self.temporal_stride
        t_idxs = tuple(zip(t0, t1))
        indices_per_sample = tuple(itertools.product(case_idxs, t_idxs))
        return indices_per_sample

    @staticmethod
    def _make_batch_indices(n_samples, batch_size):
        batch_indices = list(range(0,n_samples,batch_size)) + [n_samples]
        return tuple(zip(batch_indices[:-1], batch_indices[1:]))
            
    @staticmethod
    def _sensor_inputs_to_predicted_fields(sensor_inputs, model, bsize):
        print('Reconstructing flow fields')
        reconstructed_fields = []
        n_cases = sensor_inputs.shape[0]
        n_timesteps = sensor_inputs.shape[1]
        n_sensors = sensor_inputs.shape[2]
        batch_indices = TimeMarchingDataset._make_batch_indices(n_cases*n_timesteps, bsize)

        reshaped_sensor_inputs = sensor_inputs.reshape([-1,n_sensors])
        
        for batch_start,batch_end in tqdm(batch_indices):
            inp = reshaped_sensor_inputs[batch_start:batch_end]
            pred = model(inp)
            reconstructed_fields.append(pred.numpy())
        
        reconstructed_fields = np.reshape(np.concatenate(reconstructed_fields,0), [n_cases, n_timesteps] + list(reconstructed_fields[0].shape[1:]))

        return reconstructed_fields

    def shuffle(self):
        self._case_temporal_indices = list(self._case_temporal_indices)
        random.shuffle(self._case_temporal_indices)
        self._case_temporal_indices = tuple(self._case_temporal_indices)

    @classmethod
    def from_hdf5(cls, h5path, batch_size=1, train_test_split = None, train_test_split_type = None, rand_seed=None, random_split=False, return_sensor_values = False, return_normalization_parameters = False, return_case_ids = False, return_case_names = False, temporal_stride=1, dataset_kwargs = None, **extraction_kwargs):

        if dataset_kwargs is None:
            dataset_kwargs = {}
            
        extracted_values = _extract_values(h5path, retain_time_dimension=True, **extraction_kwargs)
        case_id_map = extracted_values[5]

        if rand_seed is not None:
            np.random.seed(rand_seed)

        if train_test_split is not None:
            assert ((1.0-train_test_split) > 0.0)
            split_axis = cls._train_test_split_method_axes[train_test_split_type]
            train_arrays, test_arrays = split_arrays(train_test_split,split_axis,*extracted_values[:5], shuffle=random_split)
            train_init_arrays = _pick_init_arrays(train_arrays,
                                                return_normalization_parameters=return_normalization_parameters,
                                                return_case_names=return_case_names,
                                                return_case_ids=return_case_ids)
            test_init_arrays = _pick_init_arrays(test_arrays,
                                                return_normalization_parameters=return_normalization_parameters,
                                                return_case_names=return_case_names,
                                                return_case_ids=return_case_ids)
            train_dset = cls(batch_size, temporal_stride=temporal_stride, **train_init_arrays, **dataset_kwargs)
            test_dset = cls(batch_size, temporal_stride=temporal_stride, **test_init_arrays, **dataset_kwargs)
            return train_dset, test_dset, case_id_map
        else:
            train_init_arrays = _pick_init_arrays(*extracted_values[:5],
                                                return_normalization_parameters=return_normalization_parameters,
                                                return_case_names=return_case_names,
                                                return_case_ids=return_case_ids)
            train_dset = cls(batch_size, temporal_stride=temporal_stride, **train_init_arrays, **dataset_kwargs)
            return train_dset, case_id_map

    def __getitem__(self, idx):
        batch_start, batch_end = self._indices_per_batch[idx]
        sample_indices = self._case_temporal_indices[batch_start:batch_end]
        sample_indices = [[x[0],np.array([x[1][0],x[1][1]])] for x in sample_indices]
        
        return_tensors = [[] for _ in range(len(self._return_order))]
        for sample in sample_indices:
            for rt, tn in zip(return_tensors, self._return_order):
                tensor = getattr(self, tn)[sample[0],sample[1]]
                if (tn == 'full_field_values') and (self._reconstructed_fields is not None) and (np.random.rand() > self.p_gt_substitute):
                    tensor[0,...] = self._reconstructed_fields[sample[0], sample[1][0]]
                rt.append(tensor)
        
        return_tensors = [np.stack(x,0).astype(tf.keras.backend.floatx()) if (x[0].dtype in self._float_dtypes) else np.stack(x,0) for x in return_tensors]
        return_tensors = [return_tensors[0][:,0], return_tensors[0][:,1]] + return_tensors[1:]

        return tuple(return_tensors)

if __name__ == '__main__':

    #def dummy_model(inpshape)
    
    trd, ted, _ = TimeMarchingDataset.from_hdf5('/storage/pyfr/big_bezier_dataset/annulus_medium_64x256.h5', batch_size = 10, train_test_split=0.8, random_split=True, reshape_to_grid=True, retain_variable_dimension=True, normalization = 'full_field_mean_center', return_normalization_parameters=True, temporal_stride=3)
    trd.shuffle()
    _ = trd[0]

    import time
    t0 = time.time()
    b = trd[1]
    t1 = time.time()
    print(f'Time taken: {t1-t0}')

    import pdb; pdb.set_trace()

    
