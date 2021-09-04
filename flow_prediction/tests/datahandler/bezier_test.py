from ...data.geometry.bezier_shapes.shapes_utils import Shape

shapeopt = {'n_control_pts': 3, 'n_sampling_pts': 10, 'radius': [0.5], 'edgy': [1.0], 'control_pts': None}
meshopt = {'mesh_domain': True, 'xmin': -10.0, 'xmax': 10.0, 'ymin': -10.0, 'ymax': 10.0, 'element_scaler': 10, 'wake_refined': True}
magnify=1.5
s = Shape(**shapeopt)
s.generate(magnify=magnify)
meshfname, n_eles = s.mesh(**meshopt)
print(f'Mesh path: {meshfname} | No of elements: {n_eles}')
print('This test should generate c. 100-150k elements')
