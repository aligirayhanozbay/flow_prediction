import os
import subprocess
import concurrent.futures
import itertools
import subprocess
import numpy as np
import h5py
import multiprocessing
import glob
import warnings
import copy

from ..utils.PyFRIntegratorHandler import PyFRIntegratorHandler
from ..geometry.bezier_shapes.shapes_utils import Shape

def shell_cmd_stdout(cmd):
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).communicate()[0]

class BasePyFRDatahandler:
    bezier_default_opts = {
        'shape':{
            'control_pts': None,
            'n_control_pts': 6,
            'n_sampling_pts': 40,
            'radius': [0.5],
            'edgy': [1.0]
        },
        'mesh':{
            'mesh_domain': True,
            'xmin': -10.0,
            'xmax': 25.0,
            'ymin': -10.0,
            'ymax': 10.0
        },
        'generate':{
            'magnify': 1.0,
            'cylinder': False,
            'centering': True,
            'ccws': True
        }
    }
    device_placement_counter = itertools.count()
    def __init__(self, mesh_files = None, case_ids = None, pyfr_configs = None, n_bezier = 0, bezier_config = None, bezier_case_prefix = None, load_cached = False, force_remesh = False, keep_directories_clean = True, backend = None, auto_device_placement = False, sample_times = None, residuals_folder = None, residuals_frequency = 20, bezier_mapping_quality_threshold = np.inf, save_dtype = 'float64', forces_folder = None, forces_frequency = 20):
        '''
        -mesh_files: List[str]. Paths to .geo/.msh/.pyfrm files.
        -case_ids: List[str]. Case ID to assign to each solver in mesh_files. 
        -pyfr_configs: str or List[str]. Path(s) to PyFR .ini file(s).
        -n_bezier: int. Number of randomly generated Bezier geometry cases to generate and run.
        -bezier_config: Dict. Config to generate the Bezier geometries. bezier_config['shape'] are the kwargs for the Shape object, bezier_config['mesh'] are the kwargs for the Shape.mesh method, bezier_config['magnify'] sets a magnification factor for the generated object inside the domain.
        -bezier_case_prefix: str. Prefix for the case names of Bezier geometries. Example: providing 'foo_' may result in a case with ID 'foo_0'
        -force_remesh: bool. If False and e.g. a .msh file corresponding to a given .geo file exists in the same directory, this mesh will be used as-is. If True, new .msh/.pyfrm will be overwritten.
        -keep_directories_clean: bool. If True, all new generated .msh/.pyfrm files will be created in /tmp/.
        -backend: str. PyFR backend to use. Defaults to openmp (i.e. CPU).
        -auto_device_placement: bool, int, List[int] or Dict[int,float]. False follows default pyfr behaviour based on the ini. True to automatically place each new created solver on different devices in a round-robin fashion. List of integers to use the specified device IDs round-robin. If a dict, the keys are the device ids to be used and the corresponding float value assigns a weight - e.g. {0:0.25, 1:0.75} with 4 solvers will assign 1 to device 0 and 3 to device 1. Does nothing for openmp backend.
        -sample_times: List of 3 floats. Start, stop and step for snapshots' times.
        -residuals_folder: str. Folder to put residual information in. No residuals will be written out if left as None.
        -residuals_frequency: int. # of timesteps per residual output.
        -forces_folder: str. Folder to put calculated forces in. Forces will be written out if left as None.
        -forces_frequency: int. # of timesteps per forces output.
        -bezier_mapping_quality_threshold: float. 
        -save_dtype: str - "float16", "float32" or "float64".
        '''
        backend = 'openmp' if backend is None else backend
        self.save_dtype = save_dtype
        self.force_remesh = force_remesh
        self.keep_directories_clean = keep_directories_clean
        self.residuals_folder = residuals_folder
        self.residuals_frequency = residuals_frequency
        self.forces_folder = forces_folder
        self.forces_frequency = forces_frequency
        self.bezier_case_prefix = bezier_case_prefix if bezier_case_prefix is not None else 'shape_'
        self.bezier_config = self._merge_bezier_opts(bezier_config)
        self.bezier_mapping_quality_threshold = bezier_mapping_quality_threshold

        self.device_placement_map, self._device_placement_wraparound = self._get_device_placement_map(auto_device_placement, backend)

        self.sample_times = self._get_sample_times(sample_times)#keeps track of the physical times to compute the soln at
        
        self.integrators = self._get_integrators(mesh_files, pyfr_configs, case_ids, n_bezier, backend)

        self._start_integrators()

        #Preallocate solver cache
        self._preallocate_caches()

    def _merge_bezier_opts(self, user_opts):
        if user_opts is None:
            return self.bezier_default_opts
        else:
            merged_opts = {}
            for field in self.bezier_default_opts:
                merged_opts[field] = {**(self.bezier_default_opts[field]), **(user_opts.get(field, {}))}
            return merged_opts

    def _get_integrators(self, mesh_files, pyfr_configs, case_ids, n_bezier, backend):
        ###################
        #Create the actual solvers
        ###################
        integrators = []
        #integrators for specified mesh files
        mesh_files = list(itertools.chain.from_iterable([glob.glob(fname) for fname in mesh_files]))
        case_ids = self._handle_case_ids(case_ids, mesh_files)
        if mesh_files is not None and pyfr_configs is not None:
            integ_ids = [next(self.device_placement_counter) for _ in mesh_files]
            
            integs = []
            for m,c,i,cid in zip(mesh_files, itertools.repeat(pyfr_configs) if isinstance(pyfr_configs, str) else pyfr_configs, integ_ids, case_ids):
                integs.append(self.create_integrator_handler(m,c,backend,i,cid))
            integrators = integrators + list(integs)
        elif (mesh_files is not None) ^ (pyfr_configs is not None):
            raise(ValueError('Both mesh files and PyFR config files must be supplied'))
        
        #integrators for randomly generated Bezier geometry cases
        if (n_bezier > 0) and (self.bezier_config is not None):
            cfg_iterator = itertools.repeat(pyfr_configs if isinstance(pyfr_configs,str) else pyfr_configs[0], n_bezier)
            backend_iterator = itertools.repeat(backend, n_bezier)
            integ_ids = [next(self.device_placement_counter) for _ in range(n_bezier)]

            integs = []
            for c,b,i in zip(cfg_iterator, backend_iterator, integ_ids):
                integs.append(self._create_bezier_integrator(c,b,i))
            integrators = integrators + list(integs)
        return integrators

    def _start_integrators(self):
        ###################
        #Create solver child processes, without starting the solution process
        ###################
        for integ in self.integrators:
            integ.start()

    @staticmethod
    def _get_sample_times(sample_times):
        ###################
        #Times to save the solution at - for self.generate_data()
        #For now, only generation of data in advance is supported.
        ###################
        if isinstance(sample_times, list) or isinstance(sample_times, tuple):
            if isinstance(sample_times[2], float):
                sample_times[2] = round((sample_times[1] - sample_times[0])/sample_times[2])+1
            sample_times = np.linspace(*sample_times)
        else:
            sample_times = np.array(sample_times)
        return sample_times
        
    @staticmethod
    def _get_device_placement_map(auto_device_placement, backend):
        ###################
        #Auto device placement code
        ###################
        if isinstance(auto_device_placement, bool) and auto_device_placement and (not backend == 'openmp'):
            #do round-robin placement on all devices if auto_device_placement=True
            #based on the backend, get the number of devices
            if backend.lower() == 'opencl':
                n_devices = int(shell_cmd_stdout('clinfo -l | grep -i device | wc -l'))
            elif backend.lower() == 'cuda':
                n_devices = int(shell_cmd_stdout('nvidia-smi -L | wc -l'))
            elif backend.lower() == 'hip':
                n_devices = int(shell_cmd_stdout('rocm-smi --showserial | grep -i "\[[0-9].*\]" | wc -l'))
            #create list of all devices and proceed as if a list was supplied instead
            auto_device_placement = list(range(n_devices))
        if (isinstance(auto_device_placement, list) or isinstance(auto_device_placement, tuple)) and (not backend == 'openmp'):
            #if devices are given as a list, convert them into a dict with same weight for each device
            #proceed as if a dict was supplied instead
            n_devices = len(auto_device_placement)
            auto_device_placement = {str(dev_index): 1/n_devices for dev_index in range(n_devices)}
        if isinstance(auto_device_placement, dict) and (not backend == 'openmp'):
            #build a dict mapping the modulo of the solver object index to the device id
            min_weight = min(auto_device_placement.values())
            auto_device_placement = {int(dev_index): round(dev_weight/min_weight) for dev_index,dev_weight in zip(auto_device_placement.keys(), auto_device_placement.values())}
            solvers_per_round = np.array(sorted([(dev_index, solvers_per_round) for dev_index, solvers_per_round in zip(auto_device_placement.keys(), auto_device_placement.values())], key=lambda x: x[0]))
            solvers_per_round_cumsum = np.concatenate([np.array([0]), np.cumsum(solvers_per_round[:,1])],0)
            auto_device_placement_bounds_by_device = np.zeros((solvers_per_round.shape[0],3), dtype=np.int64)
            auto_device_placement_bounds_by_device[:,0] = solvers_per_round[:,0]
            auto_device_placement_bounds_by_device[:,1] = solvers_per_round_cumsum[:-1]
            auto_device_placement_bounds_by_device[:,2] = solvers_per_round_cumsum[1:]

            device_placement_map = {}
            for device_bounds in auto_device_placement_bounds_by_device:
                device_id, min_device_modulus, max_device_modulus = device_bounds
                for modulus in range(min_device_modulus, max_device_modulus):
                    device_placement_map[modulus] = device_id

            device_placement_map = device_placement_map
            device_placement_wraparound = solvers_per_round_cumsum[-1]
        elif backend == 'openmp' or (isinstance(auto_device_placement, bool) and (not auto_device_placement)):
            device_placement_map = None
            device_placement_wraparound = None
        else:
            raise(ValueError('Error parsing device placement map for auto device placement. auto_device_placement must be a bool, List[int] or Dict[int,float]. You supplied: ' + str(auto_device_placement)))
        return device_placement_map, device_placement_wraparound

    @staticmethod
    def _handle_case_ids(case_ids, mesh_files):
        if case_ids is None:
            _basenames_noext = list(map(lambda x: os.path.splitext(os.path.basename(x))[0], mesh_files))
            if len(list(set(_basenames_noext))) < len(_basenames_noext):
                warnings.warn('Some mesh files have identical base names - copies will get random case ids. Remove meshes with duplicate names, or specify case_ids manually to avoid this.')
                encountered_names = []
                new_case_ids = []
                for basename in _basenames_noext:
                    if basename in new_case_ids:
                        new_case_ids.append(None)
                    else:
                        new_case_ids.append(basename)
                        encountered_names.append(basename)
            else:
                new_case_ids = _basenames_noext
            return new_case_ids
        else:
            return case_ids

    def __del__(self):
        for integ in self.integrators:
            integ.stop()

    def _geo_to_msh(self,path):
        path = os.path.realpath(path)
        foldername = os.path.dirname(path)
        filename, ext = os.path.splitext(path)
        msh_filename = filename + '.msh'
        if (not self.force_remesh) and os.path.exists(msh_filename):
            return msh_filename
        else:
            if self.keep_directories_clean:
                msh_filename = '/tmp/' + os.path.basename(msh_filename)
            r = subprocess.Popen(['gmsh', '-2',path,'-o',msh_filename])
            return msh_filename

    def _msh_to_pyfrm(self,path):
        path = os.path.realpath(path)
        filename,ext = os.path.splitext(path)
        pyfrm_path = filename + '.pyfrm'
        if (not self.force_remesh) and os.path.exists(pyfrm_path):
            return pyfrm_path
        else:
            if self.keep_directories_clean:
                pyfrm_path = '/tmp/' + os.path.basename(pyfrm_path)
            r = shell_cmd_stdout(f'pyfr import -tgmsh {path} {pyfrm_path}')
            return pyfrm_path

    def _handle_mesh_file(self,mesh_file_path):
        '''
	Helper function to convert .geo and .msh files to .pyfrm if needed.
        '''
        if 'geo' in os.path.splitext(mesh_file_path)[1]:
            mesh_file_path = self._geo_to_msh(mesh_file_path)
        if ('msh' in os.path.splitext(mesh_file_path)[1]) or ('mesh' in os.path.splitext(mesh_file_path)[1]):
            mesh_file_path = self._msh_to_pyfrm(mesh_file_path)
        return mesh_file_path

    def create_integrator_handler(self,mesh, config_path, backend, integrator_id = None, case_id = None):
        if self.device_placement_map is not None:
            integrator_id = integrator_id if integrator_id is not None else next(self.device_placement_counter)
            device_id = self.device_placement_map[integrator_id % self._device_placement_wraparound]
        else:
            device_id = None
        mesh_file_path = self._handle_mesh_file(mesh)
        #residuals_path = self.residuals_folder + '/' + case_id + '_residuals.csv' if self.residuals_folder is not None else None
        return PyFRIntegratorHandler(mesh_file_path, config_path, backend = backend, device_id = device_id, case_id = case_id, residuals_path = self.residuals_folder, residuals_frequency = self.residuals_frequency, forces_path = self.forces_folder, forces_frequency = self.forces_frequency)

    def _create_bezier_integrator(self, config_path, backend, integrator_id = None):
        
        if self.bezier_mapping_quality_threshold is not None:
            max_tries = 25
            check_mapping = [self.bezier_config['mesh'].get('xmin', -1.0),
                             self.bezier_config['mesh'].get('xmax', 1.0),
                             self.bezier_config['mesh'].get('ymin', -1.0),
                             self.bezier_config['mesh'].get('ymax', 1.0)]
            generate_opts = copy.deepcopy(self.bezier_config['generate'])
            generate_opts['check_mapping'] = check_mapping
            for i in range(max_tries):
                shape = Shape(**self.bezier_config['shape'])
                qual = shape.generate(**generate_opts)
                if (qual[0] < self.bezier_mapping_quality_threshold) and (qual[1] < self.bezier_mapping_quality_threshold):
                    print(qual)
                    break
                if i == (max_tries-1):
                    raise(RuntimeError('Could not create Bezier shape with sufficient mapping quality.'))
        else:
            shape = Shape(**self.bezier_config['shape'])
            _ = shape.generate(**self.bezier_config['generate'])
        mesh_file_path, n_eles = shape.mesh(**self.bezier_config['mesh'])
        print(f'Meshed {mesh_file_path} with {n_eles} elements')
        case_id = self.bezier_case_prefix + str(shape.index)
        return self.create_integrator_handler(mesh_file_path, config_path, backend, integrator_id = integrator_id, case_id = case_id)

    def advance_to(self, t):
        for integ in self.integrators:
            integ.advance_to(t)

    @staticmethod
    def _get_snapshot(integ, concatenated=True):
        #overload this to record different quantities
        solns = integ.concatenated_soln(gradients=False).transpose(1,0)
        grads = integ.concatenated_soln(gradients=True).transpose(1,0)
        return np.concatenate([solns,grads],-1) if concatenated else (solns, grads)
        

    #@staticmethod
    def _get_snapshot_wrap(self, integ, queue):
        snapshot = self._get_snapshot(integ)
        queue.put((integ.case_id,snapshot))

    def _place_into_cache(self, tidx, r):
        self.cache[r[0]][tidx] = r[1]

    def _record_snapshots(self, tidx):
        q = multiprocessing.Queue()
        processes = []
        for idx, integ in enumerate(self.integrators):
            p = multiprocessing.Process(target=self._get_snapshot_wrap, args=[integ, q])
            p.start()
            processes.append(p)
        for p in processes:
            r = q.get()
            self._place_into_cache(tidx, r)
        for p in processes:
            p.join()
        
    def generate_data(self):
        for tidx, t in enumerate(self.sample_times):
            self.advance_to(t)
            self._record_snapshots(tidx)
                
    def save(self, path = None):
        if path is None:
            path = 'results.h5'
        with h5py.File(path, 'w') as f:
            for integ in self.integrators:
                subgroup = f.create_group(integ.case_id)
                subgroup.create_dataset('solution',
                                        data=self.cache[integ.case_id][...,:integ.n_solnvars],
                                        dtype=self.save_dtype
                                        )
                subgroup.create_dataset('gradients',
                                        data=self.cache[integ.case_id][...,integ.n_solnvars:],
                                        dtype=self.save_dtype
                                        )
                subgroup.create_dataset('coordinates',
                                        data=integ.mesh_coords,
                                        dtype=self.save_dtype
                                        )

    def _snapshot_shape(self, integ):
        return (integ.mesh_coords_shape[0], integ.n_solnvars*(1+integ.n_spatialdims))
        
    def _compute_cache_shape(self, integ):
        return (self.sample_times.shape[0], *self._snapshot_shape(integ))

    def _preallocate_caches(self):
        self.cache = {}
        for integ in self.integrators:
            self.cache[integ.case_id] = np.zeros(self._compute_cache_shape(integ)).astype(self.save_dtype)

    def __getitem__(self, idx=None):
        raise(NotImplementedError)


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
    parser.add_argument('--forces_folder', type=str, help='Folder to save calculated body forces in. If not specified, the body forces will not be recorded.', default=None)
    parser.add_argument('--forces_frequency', type=int, default=20, help='Timestep frequency for recording forces')
    parser.add_argument('--save_dtype', type=str, default='float64', help='Floating point data type for saving')
    args = parser.parse_args()

    if args.bezier_cfg is not None:
        bezier_cfg = json.load(open(args.bezier_cfg,'r'))
    else:
        bezier_cfg = None
    if len(args.pyfr_configs) == 1:
        args.pyfr_configs = args.pyfr_configs[0]

    dh = BasePyFRDatahandler(
        mesh_files = args.meshes,
        pyfr_configs = args.pyfr_configs,
        n_bezier = args.n_bezier,
        backend = args.backend,
        auto_device_placement = args.no_auto_device_placement,
        sample_times = args.sample_times,
        bezier_config = bezier_cfg,
        residuals_folder = args.residuals_folder,
        residuals_frequency = args.residuals_frequency,
        forces_folder = args.forces_folder,
        forces_frequency = args.forces_frequency,
        save_dtype = args.save_dtype
    )

    print(f'Running cases: {[i.case_id for i in dh.integrators]}')

    dh.generate_data()

    dh.save(args.save_path)

    del dh
    
    
    
    
    
