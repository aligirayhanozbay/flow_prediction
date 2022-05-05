import h5py
import numpy as np
import multiprocessing as mp
import threading
import time

from .base import BasePyFRDatahandler

class SequentialPyFRDatahandler(BasePyFRDatahandler):
    
    def __init__(self, *args, save_path = None, **kwargs):
        #similar to BasePyFRdatahandler but runs cases one-by-one for more memory-intensive scenarios
        super().__init__(*args, **kwargs)
        self.save_path = save_path or 'result.h5'
        self._write_pipe_parent, self._write_pipe_child = mp.Pipe(duplex=True)

    def _start_integrators(self):
       #no-op as integrators should not be
       #started at init for this class
       pass

    def save(self, *args, **kwargs):
        raise(NotImplementedError)

    def _write_to_disk(self, tidx, case_id, solns, grads):
        if case_id in self.h5f:
            self.h5f[case_id]['solution'][tidx] = solns
            self.h5f[case_id]['gradients'][tidx] = grads
        else:
            subgroup = self.h5f.create_group(case_id)
            subgroup.create_dataset('solution',
                                    [len(self.sample_times), *solns.shape],
                                    dtype=self.save_dtype)
            subgroup.create_dataset('gradients',
                                    [len(self.sample_times), *grads.shape],
                                    dtype=self.save_dtype)

    def _writer_process(self):
        try:
            self.h5f = h5py.File(self.save_path, 'r+')
        except:
            self.h5f = h5py.File(self.save_path, 'w')

        n_integs = len(self.integrators)
        n_finished = 0
        while True:
            tidx, integ_idx = self._write_pipe_child.recv()
            integ = self.integrators[integ_idx]
            case_id = integ.case_id
            if tidx == -1:
                self.h5f.close()
                return
            else:
                grads, solns = self._get_snapshot(integ, concatenated=False)
                self._write_pipe_child.send(0)
                self._write_to_disk(tidx, case_id, solns, grads)


    def _preallocate_caches(self):
        #no caching data to ram for this class - write directly to disk
        pass

    def generate_data(self):

        device_ids = [x.device_id for x in self.integrators]
        unique_device_ids = list(set(device_ids))
        queues = [mp.Queue() for _ in unique_device_ids]
        for iidx, integ in enumerate(self.integrators):
            device_id = device_ids[iidx]
            queue_idx = unique_device_ids.index(device_id)
            queues[queue_idx].put(iidx)
            
        cur_integs = [q.get() for q in queues]
        while not np.all([(c is None) for c in cur_integs]):
            print(f'Running cases: {cur_integs}')
            for iidx in cur_integs:
               self.integrators[iidx].start()

            writer_process = mp.Process(target=self._writer_process)
            writer_process.start()
                
            for tidx, t in enumerate(self.sample_times):
                for iidx in filter(lambda x: x is not None, cur_integs):
                    self.integrators[iidx].advance_to(t)

                for iidx in filter(lambda x: x is not None, cur_integs):
                    self._write_pipe_parent.send([tidx, iidx])
                    #await dummy value for synchronization
                    _ = self._write_pipe_parent.recv()

            for iidx in cur_integs:
                self.integrators[iidx].stop()

            cur_integs = [q.get() if (not q.empty()) else None for q in queues]

            self._write_pipe_parent.send([-1,-1])
            writer_process.join()
                
        
        return

    def __del__(self):
        pass
    #     self._write_pipe_parent.send([-1,-1,-1])
        
if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser('Generate raw solution data with PyFR')
    parser.add_argument('--save_path', type=str, help='path to the output file')
    parser.add_argument('--pyfr_configs', nargs = '*', type=str, help='config(s) for the pyfr simulations. provide 1 config to use with all simulations.')
    parser.add_argument('--meshes', nargs='*', type=str, help='paths to the mesh files', default=[])
    parser.add_argument('--n_bezier', type=int, default=0, help='# of randomly generated bezier geometries')
    parser.add_argument('--bezier_cfg', type=str, default=None, help='path to a json file conatining the config for bezier shape generation and meshing')
    parser.add_argument('--backend', type=str, default='openmp', help="'cuda', 'opencl', 'openmp' or 'hip'")
    parser.add_argument('--sample_times', nargs=3, type=float, help='3 floats for start stop step', default = [5.0,120.0,0.25])
    parser.add_argument('--no-auto-device-placement', action='store_false')
    parser.add_argument('--residuals_folder', type=str, help='Folder to save solution residuals in. If not specified, no residuals will be recorded.', default=None)
    parser.add_argument('--residuals_frequency', type=int, default=20, help='Timestep frequency for recording residuals')
    parser.add_argument('--mapping_quality', type=float, default=np.inf, help='Threshold for re-attempting mapping if map quality is low.')
    parser.add_argument('--save_dtype', type=str, default='float64', help='Floating point data type for saving')
    args = parser.parse_args()

    if args.bezier_cfg is not None:
        bezier_cfg = json.load(open(args.bezier_cfg,'r'))
    else:
        bezier_cfg = None
    if len(args.pyfr_configs) == 1:
        args.pyfr_configs = args.pyfr_configs[0]

    dh = SequentialPyFRDatahandler(
        mesh_files = args.meshes,
        pyfr_configs = args.pyfr_configs,
        n_bezier = args.n_bezier,
        backend = args.backend,
        auto_device_placement = args.no_auto_device_placement,
        sample_times = args.sample_times,
        bezier_config = bezier_cfg,
        residuals_folder = args.residuals_folder,
        residuals_frequency = args.residuals_frequency,
        bezier_mapping_quality_threshold = args.mapping_quality,
        save_path = args.save_path,
        save_dtype = args.save_dtype
    )

    print(f'Running cases: {[i.case_id for i in dh.integrators]}')

    dh.generate_data()

    del dh
