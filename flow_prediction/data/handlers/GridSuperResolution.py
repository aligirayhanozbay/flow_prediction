from .base import BasePyFRDatahandler
import numpy as np
import h5py

class GridSuperResolution(BasePyFRDatahandler):

    def __init__(self, batch_size, grid_extent, high_res_gridsize, downsample_factor, inputs_retain_shape = False, load_cache = None, **base_args):

        self.batch_size = batch_size
        self.inputs_retain_shape = inputs_retain_shape #if true e.g. [2,3,4,5,6,7] is downsampled to [2,2,4,4,6,6]
        

        self.high_res_grid_coords = self._generate_grid(grid_extent, high_res_gridsize)
        self.high_res_gridsize = high_res_gridsize

        downsample_factor = [int(downsample_factor) for _ in self.high_res_gridsize] if (not hasattr(downsample_factor, '__getitem__')) else [int(downsample_factor) for d in downsample_factor]
        self._downsampling_slice = tuple([Ellipsis] + [slice(None,None,d) for d in downsample_factor])

        super().__init__(**base_args)
        #self.low_res_grid_coords = self._generate_grid(grid_extent, low_res_gridsize)
        #self.low_res_gridsize = low_res_gridsize
        #self.low_res_cache = np.zeros([len(self.integrators), 0, *self.low_res_gridsize])

    @staticmethod
    def _generate_grid(extent, size):
        '''
        -extent: List[List[float,float]]. 'Corners' of the rectangle to get the samples from.
        -size: List[int]. Number of gridpoints per dimension.

        Returns:
        -coords: np.array with shape [np.prod(size), len(size)]. Coordinates of each point on the grid.
        '''
        return np.reshape(np.stack(np.meshgrid(*[np.linspace(g[0],g[1],h) for g,h in zip(extent, size)], indexing='ij'),-1), [-1, len(size)])

    @staticmethod
    def _grads_to_vorticity(grads):
        return grads[..., 2, 0] - grads[..., 1, 1]

    def _downsample(self, inp):
        downsampled_inp = inp[self._downsampling_slice]
        if self.inputs_retain_shape:
            for dim in range(2,2+len(self.high_res_gridsize)):
                downsampled_inp = downsampled_inp.repeat(self._downsampling_slice[-1].step, axis=dim)
        return downsampled_inp

    def _snapshot_shape(self, integ=None):
        return self.high_res_gridsize

    def _record_snapshots(self, tidx):
        for integ in self.integrators:
            self.cache[integ.case_id][tidx] = self._grads_to_vorticity(integ.interpolate_soln(self.high_res_grid_coords, gradients = True)).reshape(*self.high_res_gridsize)

    def save(self, path = None):
        if path is None:
            path = 'results.h5'
        with h5py.File(path, 'w') as f:
            for integ in self.integrators:
                subgroup = f.create_group(integ.case_id)
                subgroup.create_dataset('vorticity', data=self.cache[integ.case_id])
                subgroup.create_dataset('coordinates', data=self.high_res_grid_coords)

    def __getitem__(self, idx):
        pass
            
    
