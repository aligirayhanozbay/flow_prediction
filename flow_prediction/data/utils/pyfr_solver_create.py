from pyfr.backends import get_backend
from pyfr.rank_allocator import get_rank_allocation
from pyfr.solvers import get_solver
from pyfr.progress_bar import ProgressBar
from pyfr.mpiutil import register_finalize_handler

import os

def pyfr_solver_create(mesh, soln, cfg, backend = 'openmp', progress_bars = True):
        from dataclasses import dataclass
        @dataclass
        class args_class:
                progress: bool
                backend: str
        args = args_class(progress_bars, backend)

        # Prefork to allow us to exec processes after MPI is initialised
        if hasattr(os, 'fork'):
                from pytools.prefork import enable_prefork

        enable_prefork()

        # Import but do not initialise MPI
        from mpi4py import MPI

        # Manually initialise MPI
        if not MPI.Is_initialized():
                MPI.Init()

        # Ensure MPI is suitably cleaned up
        register_finalize_handler()

        # Create a backend
        backend = get_backend(args.backend, cfg)

        # Get the mapping from physical ranks to MPI ranks
        rallocs = get_rank_allocation(mesh, cfg)

        # Construct the solver
        solver = get_solver(backend, rallocs, mesh, soln, cfg)

        # If we are running interactively then create a progress bar
        if args.progress and MPI.COMM_WORLD.rank == 0:
                pb = ProgressBar(solver.tstart, solver.tcurr, solver.tend)

                # Register a callback to update the bar after each step
                callb = lambda intg: pb.advance_to(intg.tcurr)
                solver.completed_step_handlers.append(callb)

        return solver
