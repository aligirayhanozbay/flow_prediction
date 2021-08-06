from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
import multiprocessing
import concurrent.futures
import scipy.interpolate
import numpy as np
import secrets

from .pyfr_solver_create import pyfr_solver_create

class PyFRIntegratorHandler:
        executor = concurrent.futures.ThreadPoolExecutor()
        def __init__(self, mesh_path, ini_path, backend = 'openmp', device_id=None, case_id=None):
                '''
        	Handler object to manage running multiple PyFR integrators (i.e. solvers) simultaneously.

        	Typical use case: Initialize a handler, call the start() method, call advance_to(desired physical time), get solution using the soln property or interpolate solution directly using the interpolate_soln method.

        	Init args:
        	-mesh path: str. Path to a pyfr mesh (.pyfrm) file.
        	-ini_path: str. Path to a pyfr config (.ini) file.
        	-backend: str. PyFR backend to use (openmp etc).
        	-device_id: int. id of the device to run the solver on. No effect for openmp backend.
        	-case_id: str. Unique identifier for the case/solver. Will be randomly generated if not provided.
        	'''
                self._soln = None
                self._process = None
                self._mesh_coords = None
                self._mesh_coords_shape = None
                self._n_spatialdims = None
                self._n_solnvars = None
                self.solver = None
                self.mesh_path = mesh_path
                self.ini_path = ini_path
                self.backend = backend
                self.device_id = device_id
                self._pipe_main, self._pipe_child = multiprocessing.Pipe(duplex=True)
                self.case_id = case_id if case_id is not None else secrets.token_hex(4)

        def _raise_multiprocessing_error(self):
                if (multiprocessing.current_process().name == 'MainProcess') and self.solver is None and self._process is None:#no solver available
                        raise(RuntimeError('There is no solver on the main process and no child process exists (i.e. there is no solver). Run the create_solver method to create a solver on the main thread or the start method to create a solver on a child process.'))
                elif (multiprocessing.current_process().name == 'MainProcess') and self.solver is not None and self._process is not None:
                        raise(RuntimeError('Both a child solver process and a solver on the main process exist. Something went wrong - did you call both start() and create_solver() on the main process?'))

        def _message_to_child_noreply(self,msg):
                self._pipe_main.send(msg)

        def _message_to_child_awaitreply(self,msg):
                self._pipe_main.send(msg)
                return self._pipe_main.recv()

        def create_solver(self):
                '''
        	Creates the solver object. If this is called directly, it will create the solver directly on the main process (so no concurrent/parallel execution of multiple solvers!)
                '''
                ini = Inifile.load(self.ini_path)
                mesh = NativeReader(self.mesh_path)

                if self.device_id is not None and self.backend != 'openmp':
                        ini.set(f'backend-{self.backend}', 'device-id', self.device_id)
                
                self.solver = pyfr_solver_create(mesh, None, ini, self.backend)

        def handle_message(self, msg):
                #messages: tuples - msg[0] is the action, msg[1] contains the parameters to execute the action corresponding to the message
                if msg[0] == 0:#advance solver to physical time
                        result = self.solver.advance_to(msg[1])
                elif msg[0] == 1:#interpolate soln
                        result = self._interpolate_soln(*msg[1][:-1],**msg[1][-1])
                        self._pipe_child.send(result)
                elif msg[0] == 101:#synchornize solution variable
                        self._pipe_child.send(self.soln)
                elif msg[0] == 102:#synchronize solution grad variable
                        self._pipe_child.send(self.grad_soln)
                elif msg[0] == 103:#synchronize solution mesh coordinates
                        self._pipe_child.send(self.mesh_coords)
                elif msg[0] == 104:#get mesh coords shape without the need to copy mesh coords
                        self._pipe_child.send(self.mesh_coords_shape)
                elif msg[0] == 105:#number of spatial dims
                        self._pipe_child.send(self.n_spatialdims)
                elif msg[0] == 106:#number of spatial dims
                        self._pipe_child.send(self.n_solnvars)

        def _solver_process(self):
                self.create_solver()
                while True:
                        msg = self._pipe_child.recv()
                        result = self.handle_message(msg)
                        #self._pipe_child.send(result)

        def start(self):
                '''
        	Starts a new process and creates the solver object on the child process, but does NOT start the execution of the solver.
                '''
                if self._process is None:# or (isinstance(self._process, multiprocessing.Process) and self._process.exitcode is None):
                        self._process = multiprocessing.Process(target=self._solver_process)
                        self._process.start()
                else:
                        raise(RuntimeError('Solver process already active.'))

        def stop(self):
                if self._process is not None:
                        self._process.terminate()
                        self._process.join()

        def advance_to(self, t):
                '''
        	Advances the solver on the child process to t. Do not call, unless start() has been called.
                '''
                self._pipe_main.send((0, t))

        @property
        def soln(self):
                if self.solver is None and self._process is not None:
                        self._soln = self._message_to_child_awaitreply((101,None))
                elif self.solver is not None and (self._process is None or multiprocessing.current_process().name != 'MainProcess'):
                        #solver running in main process, or property accessed from child. directly compute result.
                        self._soln = self.solver.soln
                else:
                        self._raise_multiprocessing_error()
                return self._soln

        @property
        def grad_soln(self):
                if self.solver is None and self._process is not None:
                        self._grad_soln = self._message_to_child_awaitreply((102,None))
                elif self.solver is not None and (self._process is None or multiprocessing.current_process().name != 'MainProcess'):
                        self._grad_soln = self.solver.grad_soln
                else:
                        self._raise_multiprocessing_error()
                return self._grad_soln

        def concatenated_soln(self, gradients = False):
                '''
                Returns the solution from all element types concatenated into a single array
                '''
                if gradients:
                        concatenated_vars = zip(*[s.transpose((0,2,1,3)).reshape(self.n_spatialdims * self.n_solnvars,s.shape[1],s.shape[3]) for s in self.grad_soln])
                else:
                        concatenated_vars = zip(*[s.transpose((1,0,2)) for s in self.soln])
                soln_vals = np.stack(list(map(lambda s: np.concatenate([v.reshape(-1) for v in s],0),concatenated_vars)),0) #convert solutions from different element types into a single (n_variables,total number of soln points) shaped tensor
                return soln_vals

        def _compute_mesh_coords(self):
                ndims = self.solver.pseudointegrator.system.ele_ploc_upts[0].shape[1]
                mesh_coords = np.concatenate([ecoords.transpose((0,2,1)).reshape(-1,ndims) for ecoords in self.solver.pseudointegrator.system.ele_ploc_upts],0)
                return mesh_coords

        @property
        def mesh_coords(self):
                if self._mesh_coords is None and self.solver is None and self._process is not None:
                        self._mesh_coords = self._message_to_child_awaitreply((103,None))
                elif self._mesh_coords is None and self.solver is not None and (self._process is None or multiprocessing.current_process().name != 'MainProcess'):
                        self._mesh_coords = self._compute_mesh_coords()
                elif (multiprocessing.current_process().name == 'MainProcess') and not ((self.solver is None)^(self._process is None)):
                        self._raise_multiprocessing_error()
                return self._mesh_coords

        @property
        def mesh_coords_shape(self):
                '''
                Get the shape of mesh_coords without the need to copy mesh_coords from the child process to the parent
                '''
                if self._mesh_coords_shape is None and self.solver is None and self._process is not None:
                        self._mesh_coords_shape = self._message_to_child_awaitreply((104,None))
                elif self._mesh_coords_shape is None and self.solver is not None and (self._process is None or multiprocessing.current_process().name != 'MainProcess'):
                        self._mesh_coords_shape = self.mesh_coords.shape
                elif (multiprocessing.current_process().name == 'MainProcess') and not ((self.solver is None)^(self._process is None)):
                        self._raise_multiprocessing_error()
                return self._mesh_coords_shape

        @property
        def n_spatialdims(self):
                if self._n_spatialdims is None and self.solver is None and self._process is not None:
                        self._n_spatialdims = self._message_to_child_awaitreply((105,None))
                elif self._n_spatialdims is None and self.solver is not None and (self._process is None or multiprocessing.current_process().name != 'MainProcess'):
                        self._n_spatialdims = self.solver.grad_soln[0].shape[0]
                elif (multiprocessing.current_process().name == 'MainProcess') and not ((self.solver is None)^(self._process is None)):
                        self._raise_multiprocessing_error()
                return self._n_spatialdims

        @property
        def n_solnvars(self):
                if self._n_solnvars is None and self.solver is None and self._process is not None:
                        self._n_solnvars = self._message_to_child_awaitreply((106,None))
                elif self._n_solnvars is None and self.solver is not None and (self._process is None or multiprocessing.current_process().name != 'MainProcess'):
                        self._n_solnvars = self.solver.soln[0].shape[1]
                elif (multiprocessing.current_process().name == 'MainProcess') and not ((self.solver is None)^(self._process is None)):
                        self._raise_multiprocessing_error()
                return self._n_solnvars

        def _interpolate_soln(self, target_coords, gradients = False, **scipy_griddata_opts):
                '''
	        Interpolate variables (pressure etc) onto the coordinates specified in target_coords.

		gradients: bool. If False, the soln variables will be interpolated. If True, the gradients of the soln variables will be interpolated.
	        '''
                soln_vals = self.concatenated_soln(gradients=gradients)
                interpolated_values = np.stack(list(self.executor.map(lambda v: scipy.interpolate.griddata(self.mesh_coords, v, target_coords, **scipy_griddata_opts), soln_vals)),-1)
                if gradients:
                        interpolated_values = interpolated_values.reshape(-1, self.n_solnvars, self.n_spatialdims)
                return interpolated_values

        def interpolate_soln(self, target_coords, gradients = False, **scipy_griddata_opts):
                '''
		Wraps self._interpolate_soln to work with multiprocessing
		'''
                if self.solver is None and self._process is not None:#solver process running in child process. send signal id 1 to compute interpolated vals, receive them back and return
                        return self._message_to_child_awaitreply((1,(target_coords, gradients, scipy_griddata_opts)))
                elif self.solver is not None and self._process is None:#solver running in main process. directly compute result.
                        return self._interpolate_soln(target_coords, gradients, **scipy_griddata_opts)
                else:
                        self._raise_multiprocessing_error()


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser('run the cylinder case')
	parser.add_argument('--mesh', default='./inc_cylinder_2d.pyfrm')
	parser.add_argument('--config', default='./inc_cylinder_2d.ini')
	parser.add_argument('--backend', default='opencl')
	parser.add_argument('--concurrent_solvers', type=int, default=5)
	parser.add_argument('--run_until', type=float, default=2.5)
	args = parser.parse_args()

	n_children = args.concurrent_solvers
	handlers = [PyFRIntegratorHandler(args.mesh, args.config, args.backend) for _ in range(n_children)]

	for h in handlers:
		h.start()
	for h in handlers:
		h.advance_to(args.run_until)

	ds = PyFRIntegratorHandler(args.mesh, args.config, args.backend)
	ds.create_solver()
	ds.solver.advance_to(args.run_until)

	n_plotpts_per_dim = 500
	plot_ext = [[np.min(ds.mesh_coords[:,0]), np.max(ds.mesh_coords[:,0])], [np.min(ds.mesh_coords[:,1]), np.max(ds.mesh_coords[:,1])]]
	plot_coords = np.stack(np.meshgrid(*[np.linspace(*p,n_plotpts_per_dim) for p in plot_ext]),-1).reshape(-1,len(plot_ext))
	interp_soln = ds.interpolate_soln(plot_coords, gradients=False).reshape(n_plotpts_per_dim, n_plotpts_per_dim,3)
	interp_grads = ds.interpolate_soln(plot_coords, gradients=True).reshape(n_plotpts_per_dim, n_plotpts_per_dim,3,2)
	#import pdb; pdb.set_trace()
	##interp_grads = ds.interpolate_soln(plot_coords, gradients=True).reshape(2,3,-1)

	import matplotlib.pyplot as plt
	import itertools

	def plot_var(v, fname):
		plt.figure()
		v[np.isnan(v)] = 0.0
		plt.imshow(v, cmap = 'RdBu', extent = list(itertools.chain.from_iterable(plot_ext)))
		#plt.tricontourf(plot_coords[:,0],plot_coords[:,1],v)
		##plt.tricontourf(ds.mesh_coords[:,0], ds.mesh_coords[:,1], v)
		plt.colorbar()
		plt.savefig(fname)
		plt.close()

	soln_var_map = {0: 'p', 1: 'u', 2: 'v', 3: 'w'}
	direction_map = {0: 'x', 1: 'y', 2: 'z'}
	w_z = interp_grads[..., 2, 0] - interp_grads[..., 1, 1]
	plot_var(w_z, './wz_' + str(ds.solver.tcurr) + '.png')

	import pdb; pdb.set_trace()
