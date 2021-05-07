#!python
# create a triangulation from points
# mode:
# - grid: recursively increasing radius on a regular grid
# - delaunay_2d: create a surface using delaunay
# - delaunay_3d: create a solid around points
# - outline: create a box around points
# cell_size: (optional) grid cell size
# convert_to_triangles: convert the native faces with 4 sides to 3 sides
'''
usage: $0 input_point*csv,dxf,shp,tif,tiff mode%grid,delaunay_2d,delaunay_3d,outline cell_size=10 convert_to_triangles@ output*vtk,vtm display@
'''
'''
Copyright 2017 - 2021 Vale

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*** You can contribute to the main repository at: ***

https://github.com/pemn/vtk_triangulate_points
---------------------------------

'''
import sys
import numpy as np
import pandas as pd
import os.path
import re

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')
from _gui import usage_gui, pd_load_dataframe, pd_save_dataframe

from pd_vtk import pv_save, vtk_plot_meshes, vtk_df_to_mesh, vtk_mesh_to_df, vtk_Voxel
import vtk
import pyvista as pv
from sklearn.neighbors import RadiusNeighborsRegressor

def grid_points_2d(mesh, cell_size=10):
  grid = vtk_Voxel.from_mesh(mesh, cell_size, 2)

  cells = grid.cell_centers().points

  radius = cell_size * 0.5
  tmat = np.full(cells.shape[0], np.nan)
  print("n samples", mesh.points.size, "sample min", np.min(mesh.points[:,2]), "max", np.max(mesh.points[:,2]))
  while np.any(np.isnan(tmat)):
    # keep increasing radius until all cells have values
    radius *= 1.5
    print("RadiusNeighborsRegressor =",radius,"m")
    neigh = RadiusNeighborsRegressor(radius, 'distance')
    neigh.fit(mesh.points[:,:2], mesh.points[:,2])
    rmat = neigh.predict(cells[:,:2])
    np.putmask(tmat, np.isnan(tmat), rmat)

  print("regression min", np.min(tmat), "max", np.max(tmat))
  grid.cell_arrays['Elevation'] = tmat
  surf = grid.extract_surface()
  surf = surf.ctp()
  surf.points[:, 2] = surf.point_arrays['Elevation']
  
  return surf

def grid_points_rbf(mesh, cell_size=10, function='grid'):
  grid = vtk_Voxel.from_mesh(mesh, cell_size, 2)

  samples = mesh.points
  cells = grid.cell_centers().points

  from scipy.interpolate import Rbf
  rbfi = Rbf(samples[:,0], samples[:,1], np.zeros(samples.shape[0]), samples[:,2], function=function)
  z = rbfi(cells[:,0],cells[:,1],cells[:,2])

  grid.set_ndarray('Elevation', z)
  grid = grid.ctp()
  grid = grid.extract_surface()
  grid.points[:, 2] = grid.get_array('Elevation')
  return grid


def main(input_points, mode, cell_size, convert_to_triangles, output, display):
  df = pd_load_dataframe(input_points)
  mesh = vtk_df_to_mesh(df)
  if not cell_size:
    cell_size = 10
  mesh = mesh.elevation()

  if mode == 'outline':
    grid = mesh.outline(True)
  elif mode == 'delaunay_2d':
    grid = mesh.delaunay_2d()
  elif mode == 'delaunay_3d':
    grid = mesh.delaunay_3d()
  elif mode == 'grid':
    grid = grid_points_2d(mesh, float(cell_size))
    if int(convert_to_triangles):
      grid = grid.delaunay_2d()
  else:
    grid = grid_points_rbf(mesh, float(cell_size), mode)

  if re.search(r'vt.$', output, re.IGNORECASE):
    pv_save(grid, output)
  elif output:
    df = vtk_mesh_to_df(grid)
    pd_save_dataframe(df, output)

  if int(display):
    vtk_plot_meshes([mesh, grid])

if __name__=="__main__":
  usage_gui(__doc__)
