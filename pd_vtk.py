#!python
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
'''
import numpy as np
import pandas as pd
try:
  import pyvista as pv
except:
  pv = None

''' GetDataObjectType
PolyData == 0
VTK_STRUCTURED_GRID = 2
VTK_RECTILINEAR_GRID = 3
VTK_UNSTRUCTURED_GRID = 4
UniformGrid == 6
VTK_MULTIBLOCK_DATA_SET = 13
'''

def pv_read(fp):
  ''' simple import safe pyvista reader '''
  if pv is None: return
  return pv.read(fp)

def pv_save(meshes, fp):
  ''' simple import safe pyvista writer '''
  if pv is None: return
  if not isinstance(meshes, list):
    meshes.save(fp)
  elif len(meshes) == 1:
    meshes[0].save(fp)
  else:
    pv.MultiBlock(meshes).save(fp)


def vtk_cells_to_flat(faces):
  r = []
  p = 0
  while p < len(faces):
    n = faces[p]
    r.extend(faces[p+1:p+1+n])
    p += n + 1
  return r

def vtk_flat_to_faces(flat):
  n = 0
  faces = []
  for i in range(len(flat)-1,-1,-1):
    n += 1
    faces.insert(0, i)
    if flat[i] == 0:
      faces.insert(0, n)
      n = 0
  return np.array(faces)

def pd_detect_xyz(df, z = True):
  xyz = None
  dfcs = set(df.columns)
  for s in [['x','y','z'], ['midx','midy','midz'], ['xworld','yworld','zworld'], ['xc','yc','zc']]:
    if z == False:
      s.pop()
    for c in [str.lower, str.upper,str.capitalize]:
      cs = list(map(c, s))
      if dfcs.issuperset(cs):
        xyz = cs
        break
    else:
      continue
    # break also the outter loop if the inner loop ended due to a break
    break
  if xyz is None and z:
    return pd_detect_xyz(df, False)
  return xyz

def vtk_nf_to_mesh(nodes, faces):
  if len(nodes) == 0:
    return pv.PolyData()
  if len(faces) == 0:
    return pv.PolyData(np.array(nodes))
  meshfaces = np.hstack(np.concatenate((np.full((len(faces), 1), 3, dtype=np.int_), faces), 1))
  return pv.PolyData(np.array(nodes), meshfaces)

def vtk_df_to_mesh(df, xyz = None):
  if pv is None: return
  if xyz is None:
    xyz = pd_detect_xyz(df)
  if xyz is None:
    print('geometry/xyz information not found')
    return None
  print("xyz:",','.join(xyz))
  if len(xyz) == 2:
    xyz.append('z')
    if 'z' not in df:
      if '0' in df:
        # geotiff first/only spectral channel
        print('using first channel as Z value')
        df['z'] = df['0']
      else:
        print('using 0 as Z value')
        df['z'] = 0

  pdata = df[xyz].dropna(0, 'all')
  pdata.fillna(0, inplace=True)
  pdata = pdata.values.astype(np.float)
  if 'n' in df and df['n'].max() > 0:
    mesh = pv.PolyData(pdata, vtk_flat_to_faces(df['n']))
  else:
    mesh = pv.PolyData(pdata)
  return mesh

# dmbm_to_vtk
def vtk_dmbm_to_ug(df):
  ''' datamine block model to uniform grid '''
  df_min = df.min(0)
  xyzc = ['XC','YC','ZC']

  size = df_min[['XINC','YINC','ZINC']].astype(np.int_)

  dims = np.add(df_min[['NX','NY','NZ']] ,1).astype(np.int_)

  origin = df_min[['XMORIG','YMORIG','ZMORIG']]

  grid = pv.UniformGrid(dims, size, origin)
  n_predefined = 13
  vl = [df.columns[_] for _ in range(13, df.shape[1])]
  
  cv = [dict()] * grid.GetNumberOfCells()

  for i,row in df.iterrows():
    cell = grid.find_closest_cell(row[xyzc].values)
    if cell >= 0:
      cv[cell] = row[vl].to_dict()
  cvdf = pd.DataFrame.from_records(cv)
  for v in vl:
    grid.cell_arrays[v] = cvdf[v]

  return grid

def vtk_plot_meshes(meshes):
  if pv is None: return
  p = pv.Plotter()
  c = 0
  for mesh in meshes:
    if mesh is not None:
      p.add_mesh(mesh, opacity=0.5)
      c += 1
  if c:
    print("display", c, "meshes")
    p.show()

def vtk_mesh_to_df(mesh):
  df = pd.DataFrame()
  if hasattr(mesh, 'n_blocks'):
    for block in mesh:
      df = df.append(vtk_mesh_to_df(block))
  else:
    arr_n = np.zeros(mesh.n_points, dtype=np.int)
    arr_node = np.arange(mesh.n_points, dtype=np.int)
    print("GetDataObjectType", mesh.GetDataObjectType())
    # VTK_STRUCTURED_POINTS = 1
    # VTK_STRUCTURED_GRID = 2
    # VTK_UNSTRUCTURED_GRID = 4
    # 6 = UniformGrid
    # VTK_UNIFORM_GRID = 10
    if mesh.GetDataObjectType() in [2,4,6]:
      points = mesh.cell_centers().points
      arr_node = np.arange(mesh.GetNumberOfCells(), dtype=np.int)
      arr_n = np.zeros(mesh.GetNumberOfCells())
      arr_data = [pd.Series(mesh.get_array(name),name=name) for name in mesh.cell_arrays]
    else:
      arr_data = []
      # in some cases, n_faces may be > 0  but with a empty faces array
      if mesh.n_faces and len(mesh.faces):
        face_size = int(mesh.faces[0])
        arr_flat = vtk_cells_to_flat(mesh.faces)
        points = mesh.points.take(arr_flat, 0)
        arr_node = arr_node.take(arr_flat)
        arr_n = np.tile(np.arange(face_size, dtype=np.int), len(points) // face_size)
        for name in mesh.point_arrays:
          arr_data.append(pd.Series(mesh.get_array(name).take(arr_flat), name=name))
      else:
        points = mesh.points
        arr_data = [pd.Series(mesh.point_arrays[name],name=name) for name in mesh.point_arrays]
   
    df = pd.concat([pd.DataFrame(points,columns=['x','y','z']), pd.Series(arr_n,name='n'), pd.Series(arr_node,name='node')] + arr_data,1)

  return df

def vtk_mesh_info(mesh):
  print(mesh)
  #.IsA('vtkMultiBlockDataSet'):
  if hasattr(mesh, 'n_blocks'):
    for n in range(mesh.n_blocks):
      print("block",n,"name",mesh.get_block_name(n))
      vtk_mesh_info(mesh.get(n))
  else:
    for preference in ['point', 'cell', 'field']:
      arr_list = mesh.cell_arrays
      if preference == 'point':
        arr_list = mesh.point_arrays
      if preference == 'field':
        arr_list = mesh.field_arrays

      for name in arr_list:
        arr = mesh.get_array(name, preference)
        # check if this array is unicode, obj, str or other text types
        if arr.dtype.num >= 17:
          d = np.unique(arr)
        else:
          d = '{%f <=> %f}' % mesh.get_data_range(name, preference)
        print(name,preference,arr.dtype.name,d,len(arr))
    print('')
  return mesh

def vtk_array_string_to_index(mesh):
  print("converting string arrays to integer index:")
  for name in mesh.cell_arrays:
    arr = mesh.cell_arrays[name]
    if arr.dtype.num >= 17:
      print(name,"(cell)",arr.dtype)
      mesh.cell_arrays[name] = pd.factorize(arr)[0]
  for name in mesh.point_arrays:
    arr = mesh.point_arrays[name]
    if arr.dtype.num >= 17:
      print(name,"(point)",arr.dtype)
      mesh.point_arrays[name] = pd.factorize(arr)[0]
  return mesh

def vtk_info(fp):
  if pv is None: return
  return vtk_mesh_info(pv.read(fp))


class vtk_Voxel(pv.UniformGrid):
  @classmethod
  def from_bmf(cls, bm, n_schema = None):
    if n_schema is None:
      n_schema = bm.model_n_schemas()-1
    size = np.resize(bm.model_schema_size(n_schema), 3)
    dims = bm.model_schema_dimensions(n_schema)
    o0 = bm.model_schema_extent(n_schema)
    origin = np.add(bm.model_origin(), o0[:3])
    return cls(np.add(dims, 1, dtype = np.int_, casting = 'unsafe'), size, origin[:3])

  @classmethod
  def from_mesh(cls, mesh, cell_size = 10, ndim = 3):
    mesh = mesh.copy()
    if ndim == 2:
      mesh.points[:, 2] = 0

    bb = np.transpose(np.reshape(mesh.GetBounds(), (3,2)))
    
    dims = np.add(np.ceil(np.divide(np.subtract(bb[1], bb[0]), cell_size)), 3)
    if ndim == 2:
      dims[2] = 1
    origin = np.subtract(bb[0], cell_size)
    #grid = pv.UniformGrid(dims.astype(np.int), np.full(3, cell_size, dtype=np.int), origin)
    return cls(dims.astype(np.int), np.full(3, cell_size, dtype=np.int), origin)

  @classmethod
  def from_df(cls, df, cell_size, xyz = ['x','y','z']):

    bb0 = df[xyz].min()
    bb1 = df[xyz].max()
    dims = np.add(np.ceil(np.divide(np.subtract(bb1, bb0), cell_size)), 3)
    origin = np.subtract(bb0, cell_size)

    return cls(dims.astype(np.int), np.full(3, cell_size, dtype=np.int), origin)

  @property
  def shape(self):
    shape = np.subtract(self.dimensions, 1)
    return shape[shape.nonzero()]

  def get_ndarray(self, name = None, preference='cell'):
    if name is None:
      return np.ndarray(self.shape)
    return self.get_array(name, preference).reshape(self.shape)

  def set_ndarray(self, name, array, preference='cell'):
    if preference=='cell':
      self.cell_arrays[name] = array.flat
    else:
      self.point_arrays[name] = array.flat

  def GetCellCenter(self, cellId):
    return vtk_Voxel.sGetCellCenter(self, cellId)

  # DEPRECATED: use cell_centers().points
  @staticmethod
  def sGetCellCenter(self, cellId):
    cell = self.GetCell(cellId)
    bounds = np.reshape(cell.GetBounds(), (3,2))
    return bounds.mean(1)
  
