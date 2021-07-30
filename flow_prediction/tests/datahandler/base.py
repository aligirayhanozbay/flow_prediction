import pathlib
from ...data.handlers.base import BasePyFRDatahandler

current_dir = str(pathlib.Path(__file__).parent.resolve())

meshfiles = [current_dir + '/testmesh.msh', current_dir + '../../../data/geometry/standard/cylinder_half.geo', current_dir + '/testmesh.pyfrm']
config_files = current_dir + '/testconfig.ini'
n_bezier = 4
backend = 'cuda'
auto_device_placement = True

dh = BasePyFRDatahandler(mesh_files = meshfiles, config_files = config_files, n_bezier = n_bezier, backend = backend, auto_device_placement = auto_device_placement)
dh.advance_to(1.25)
solns = [integ.soln for integ in dh.integrators]
[integ.stop() for integ in dh.integrators]
