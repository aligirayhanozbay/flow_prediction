from .base import BasePyFRDatahandler
import numpy as np
import h5py

class Sensor(BasePyFRDatahandler):

    def __init__(self, batch_size, sensor_locations, load_cache = None, **base_args):

        self.batch_size = batch_size
        self.sensor_locations = sensor_locations

        super().__init__(**base_args)

    @staticmethod
    def _grads_to_vorticity(grads):
        return grads[..., 2, 0] - grads[..., 1, 1]

    def _snapshot_shape(self, integ=None):
        return self.sensor_locations.shape[:-1]

    #@staticmethod
    def _get_snapshot(self,integ):
        return self._grads_to_vorticity(integ.interpolate_soln(self.sensor_locations, gradients = True, method='cubic')).reshape(self._snapshot_shape())

    #def _record_snapshots(self, tidx):
    #    for integ in self.integrators:
    #        self.cache[integ.case_id][tidx] = 

    def save(self, path = None):
        if path is None:
            path = 'results.h5'
        with h5py.File(path, 'w') as f:
            for integ in self.integrators:
                subgroup = f.create_group(integ.case_id)
                subgroup.create_dataset('vorticity', data=self.cache[integ.case_id])
                subgroup.create_dataset('coordinates', data=self.sensor_locations)

    def __getitem__(self, idx):
        pass
            
