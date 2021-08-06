import pathlib
from ...data.handlers.base import BasePyFRDatahandler

current_dir = str(pathlib.Path(__file__).parent.resolve())

meshfiles = [current_dir + '/testmesh.msh', current_dir + '../../../data/geometry/standard/cylinder_half.geo', current_dir + '/testmesh.pyfrm']
config_files = current_dir + '/testconfig.ini'
n_bezier = 4
backend = 'cuda'
auto_device_placement = True
sample_times = [1.0,1.5,0.05]

dh = BasePyFRDatahandler(mesh_files = meshfiles, pyfr_configs = config_files, n_bezier = n_bezier, backend = backend, auto_device_placement = auto_device_placement, sample_times = sample_times)
dh.generate_data()
dh.save('/tmp/basetest.h5')
del dh
