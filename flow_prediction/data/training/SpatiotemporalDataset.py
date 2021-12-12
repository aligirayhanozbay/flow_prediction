import h5py
import numpy as np
import tensorflow as tf
import itertools
import random

from .ShallowDecoderDataset import _extract_values

def split_arrays(split_ratio, split_axis, *arrays, shuffle=False):
    nsamples = arrays[0].shape[split_axis]
    assert np.all(np.array([arr.shape[split_axis] for arr in arrays[1:]]) == nsamples)

    max_test_case_id = round(nsamples * (1.0-split_ratio))
    global_indices = np.arange(nsamples)
    if shuffle:
        np.random.shuffle(global_indices)
    train_indices = global_indices[:-max_test_case_id]
    test_indices = global_indices[-max_test_case_id:]
        
    train_arrays = []
    test_arrays = []
    for arr in arrays:
        train_arrays.append(arr.take(indices=train_indices, axis=split_axis))
        test_arrays.append(arr.take(indices=test_indices, axis=split_axis))

    return train_arrays, test_arrays

def _pick_init_arrays(array_list, return_normalization_parameters = False, return_case_ids = False, return_case_names = False):
    init_arrays = {'sensor_values': array_list[0], 'full_field_values': array_list[1]}

    if return_normalization_parameters:
        init_arrays['normalization_params'] = array_list[2]

    if return_case_names:
        init_arrays['case_names'] = array_list[3]
        
    if return_case_ids:
        init_arrays['case_ids'] = array_list[4]

    return init_arrays
        

class SpatiotemporalFRDataset(tf.keras.utils.Sequence):
    _train_test_split_method_axes = {'case':0, None:0,'time':1}
    _numeric_dtypes = [np.float16, np.float32, np.float64, np.int16, np.int32, np.int64]
    _float_dtypes = [np.float16, np.float32, np.float64]
    def __init__(self, temporal_window, batch_size, sensor_values, full_field_values, normalization_params = None, case_names = None, case_ids = None):
        super().__init__()

        self.batch_size = batch_size
        
        self._register_return_arrays(sensor_values, full_field_values, normalization_params, case_names, case_ids)

        if isinstance(temporal_window, int):
            self.temporal_window = np.array([1.0 for _ in range(temporal_window)])
        else:
            self.temporal_window = np.array(temporal_window)
        assert self.temporal_window.dtype in self._numeric_dtypes
        assert len(self.temporal_window.shape) == 1
        
            

        self._case_temporal_indices = self._generate_per_sample_case_temporal_indices()
        assert len(self._case_temporal_indices) == self.n_samples
        self._make_batch_indices()
        assert len(self._indices_per_batch) == len(self)

    def _register_return_arrays(self, sensor_values, full_field_values, normalization_params, case_names, case_ids):
        assert sensor_values.shape[:2] == full_field_values.shape[:2]
        self.sensor_values = sensor_values
        self.full_field_values = full_field_values
        return_order = ['sensor_values', 'full_field_values']
        if normalization_params is not None:
            assert sensor_values.shape[:2] == normalization_params.shape[:2]
            self.normalization_params = normalization_params
            return_order.append('normalization_params')
        if case_names is not None:
            assert sensor_values.shape[:2] == case_names.shape[:2]
            self.case_names = case_names
            return_order.append('case_names')
        if case_ids is not None:
            assert sensor_values.shape[:2] == case_ids.shape[:2]
            self.case_ids = case_ids
            return_order.append('case_ids')

        self._return_order = return_order

    @property
    def case_name_map(self):
        return {self.case_id_map[k]:k for k in self.case_names}

    @property
    def n_cases(self):
        return self.sensor_values.shape[0]

    @property
    def n_timesteps(self):
        return self.sensor_values.shape[1]

    @property
    def temporal_window_size(self):
        return self.temporal_window.shape[0]

    @property
    def n_samples(self):
        return self.n_cases * (self.n_timesteps + 1 - self.temporal_window_size)
    
    def __len__(self):
        return int(np.ceil(self.n_samples/self.batch_size))

    def _generate_per_sample_case_temporal_indices(self):
        case_idxs = np.arange(self.n_cases)
        t_idxs = tuple(zip(
            np.arange(self.n_timesteps-self.temporal_window_size+1),
            np.arange(self.temporal_window_size, self.n_timesteps+1)
        ))
        indices_per_sample = tuple(itertools.product(case_idxs, t_idxs))
        return indices_per_sample

    def _make_batch_indices(self):
        batch_indices = list(range(0,self.n_samples,self.batch_size)) + [self.n_samples]
        self._indices_per_batch = tuple(zip(batch_indices[:-1], batch_indices[1:]))

    def shuffle(self):
        self._case_temporal_indices = list(self._case_temporal_indices)
        random.shuffle(self._case_temporal_indices)
        self._case_temporal_indices = tuple(self._case_temporal_indices)
        
    @classmethod
    def from_hdf5(cls, h5path, temporal_window, batch_size=1, train_test_split = None, train_test_split_type = None, rand_seed=None, random_split=False, return_normalization_parameters = False, return_case_ids = False, return_case_names = False, dataset_kwargs = None, **extraction_kwargs):

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
            train_dset = cls(temporal_window, batch_size, **train_init_arrays, **dataset_kwargs)
            test_dset = cls(temporal_window, batch_size, **test_init_arrays, **dataset_kwargs)
            return train_dset, test_dset, case_id_map
        else:
            train_init_arrays = _pick_init_arrays(*extracted_values[:5],
                                                return_normalization_parameters=return_normalization_parameters,
                                                return_case_names=return_case_names,
                                                return_case_ids=return_case_ids)
            train_dset = cls(temporal_window, batch_size, **train_init_arrays, **dataset_kwargs)
            return train_dset, case_id_map

    def __getitem__(self, idx):
        batch_start, batch_end = self._indices_per_batch[idx]
        sample_indices = self._case_temporal_indices[batch_start:batch_end]
        
        sample_indices = [[x[0],np.array(list(range(x[1][0],x[1][1])), dtype=np.int64)] for x in sample_indices]
        return_tensors = [[] for _ in range(len(self._return_order))]
        for sample in sample_indices:
            for rt,tn in zip(return_tensors, self._return_order):
                tensor = getattr(self,tn)[sample[0],sample[1]]
                if tn == 'sensor_values':
                    tensor = np.einsum('t...,t->t...',tensor,self.temporal_window)
                rt.append(tensor)
        return_tensors = [np.stack(x,0).astype(tf.keras.backend.floatx()) if (x[0].dtype in self._float_dtypes) else np.stack(x,0) for x in return_tensors]
            
        #tf.gather_nd version - works VERY slowly
        #sample_indices = np.array([list(zip(itertools.repeat(x[0]),range(x[1][0],x[1][1]))) for x in sample_indices], dtype = np.int64)
        # return_tensors = []
        # with tf.device('cpu'):
        #     for t in self._return_order:
        #         return_tensors.append(
        #             tf.gather_nd(getattr(self, t), sample_indices)
        #         )

        return tuple(return_tensors)
            

if __name__ == '__main__':
    
    trd, ted, _ = SpatiotemporalFRDataset.from_hdf5('/storage/pyfr/big_bezier_dataset/annulus_medium_64x256.h5', [1,0,1,0,1], batch_size = 10, train_test_split=0.8, random_split=True, reshape_to_grid=True, retain_variable_dimension=True, normalization = 'full_field_mean_center', return_normalization_parameters=True)
    trd.shuffle()
    _ = trd[0]

    import time
    t0 = time.time()
    b = trd[1]
    t1 = time.time()
    print(f'Time taken: {t1-t0}')

    import pdb; pdb.set_trace()
        
