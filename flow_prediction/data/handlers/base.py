import os
import subprocess
import concurrent.futures
import itertools
import subprocess
import numpy as np

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
            'xmin': -10.0,
            'xmax': 25.0,
            'ymin': -10.0,
            'ymax': 10.0
        },
        'magnify': 1.0
    }
    device_placement_counter = itertools.count()
    def __init__(self, mesh_files = None, config_files = None, n_bezier = 0, bezier_config = None, load_cached = False, force_remesh = False, keep_directories_clean = True, backend = None, auto_device_placement = False):
        backend = 'openmp' if backend is None else backend
        self.force_remesh = force_remesh
        self.keep_directories_clean = keep_directories_clean
        self.bezier_config = self.bezier_default_opts if bezier_config is None else {**self.bezier_default_opts,**bezier_config}

        #auto device placement code
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

            self.device_placement_map = device_placement_map
            self._device_placement_wraparound = solvers_per_round_cumsum[-1]
        elif self.backend == 'openmp' or (isinstance(auto_device_placement, bool) and (not auto_device_placement)):
            self.device_placement_map = None
        else:
            raise(ValueError('Error parsing device placement map for auto device placement. auto_device_placement must be a bool, List[int] or Dict[int,float]. You supplied: ' + str(auto_device_placement)))

        self.integrators = []
        #integrators for specified mesh files
        if mesh_files is not None and config_files is not None:
            integ_ids = [next(self.device_placement_counter) for _ in mesh_files]

            integs = []
            for m,c,i in zip(mesh_files, itertools.repeat(config_files) if isinstance(config_files, str) else config_files, integ_ids):
                integs.append(self.create_integrator_handler(m,c,backend,i))
            self.integrators = self.integrators + list(integs)
        elif (mesh_files is not None) ^ (config_files is not None):
            raise(ValueError('Both mesh files and PyFR config files must be supplied'))
        #integrators for randomly generated Bezier geometry cases
        if (n_bezier > 0) and (self.bezier_config is not None):
            cfg_iterator = itertools.repeat(config_files if isinstance(config_files,str) else config_files[0], n_bezier)
            backend_iterator = itertools.repeat(backend, n_bezier)
            integ_ids = [next(self.device_placement_counter) for _ in mesh_files]

            integs = []
            for c,b,i in zip(cfg_iterator, backend_iterator, integ_ids):
                integs.append(self._create_bezier_integrator(c,b,i))
            self.integrators = self.integrators + list(integs)

        for integ in self.integrators:
            integ.start()

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

    def create_integrator_handler(self,mesh, config_path, backend, integrator_id = None):
        if self.device_placement_map is not None:
            integrator_id = integrator_id if integrator_id is not None else next(self.device_placement_counter)
            device_id = self.device_placement_map[integrator_id % self._device_placement_wraparound]
        else:
            device_id = None
        mesh_file_path = self._handle_mesh_file(mesh)
        return PyFRIntegratorHandler(mesh_file_path, config_path, backend = backend, device_id = device_id)

    def _create_bezier_integrator(self, config_path, backend, integrator_id = None):
        shape = Shape(**self.bezier_config['shape'])
        shape.generate(self.bezier_config['magnify'])
        mesh_file_path, _ = shape.mesh(mesh_domain = True,**self.bezier_config['mesh'])
        return self.create_integrator_handler(mesh_file_path, config_path, backend, integrator_id = integrator_id)

    def advance_to(self, t):
        for integ in self.integrators:
            integ.advance_to(t)
