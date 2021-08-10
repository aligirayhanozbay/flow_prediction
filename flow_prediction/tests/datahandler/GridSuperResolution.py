import pathlib
from ...data.handlers.GridSuperResolution import GridSuperResolution

current_dir = str(pathlib.Path(__file__).parent.resolve())

meshfiles = [current_dir + '/testmesh.msh', current_dir + '../../../data/geometry/standard/cylinder_half.geo']#, current_dir + '/testmesh.pyfrm']
config_files = current_dir + '/testconfig.ini'
n_bezier = 10
backend = 'cuda'
auto_device_placement = True
bsize = 10
grid_extent = [[0.75,1.5],[-0.75,0.75]]
high_res_gridsize = [128,128]
downsample_factor = 4
sample_times = [1.0,5.0,0.05]

dh = GridSuperResolution(bsize, grid_extent, high_res_gridsize, downsample_factor, sample_times = sample_times, mesh_files = meshfiles, pyfr_configs = config_files, n_bezier = n_bezier, backend = backend, auto_device_placement = auto_device_placement)
dh.generate_data()
dh.save('/tmp/basetest.h5')
del dh
