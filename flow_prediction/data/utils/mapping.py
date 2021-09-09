import numpy as np
import pydsc
import itertools
import concurrent.futures
import os

def get_curve_points(curve_name):
    import gmsh
    phys_grps = gmsh.model.getPhysicalGroups(1)
    grp_names = list(map(lambda x: gmsh.model.getPhysicalName(*x), phys_grps))
    target_grp = phys_grps[grp_names.index(curve_name)]
    lines = [[[1,idx]] for idx in gmsh.model.getEntitiesForPhysicalGroup(*target_grp)]
    return set(itertools.chain.from_iterable(map(gmsh.model.getBoundary, lines)))

def points_to_coords(pts):
    import gmsh
    return np.stack([gmsh.model.getValue(*pt,[])[:2] for pt in pts],0)

def _get_file_vertices(fname):
    import gmsh
    gmsh.initialize()
    msh_file = os.path.splitext(fname)[0] + '.msh'
    gmsh.merge(msh_file)
    inner_pts = get_curve_points('obstacle')
    outer_pts = [[0,1],[0,2],[0,3],[0,4]]
    inner_coords = points_to_coords(inner_pts)
    inner_coords_complex = inner_coords[:,0] + 1j*inner_coords[:,1]
    outer_coords = points_to_coords(outer_pts)
    outer_coords_complex = outer_coords[:,0] + 1j*outer_coords[:,1]
    gmsh.finalize()
    return outer_coords_complex, inner_coords_complex
    
def get_vertices(paths):
    
    mesh_filenames = paths
    with concurrent.futures.ProcessPoolExecutor() as executor:
        vertices = list(executor.map(_get_file_vertices, mesh_filenames))
    return vertices

def get_maps(vertices):
    amap = pydsc.AnnulusMap(*vertices, nptq = 64)
    return amap
