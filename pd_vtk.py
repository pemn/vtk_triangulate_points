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
import re, sys, os
import logging
logging.basicConfig(format='%(message)s', level=99)
log = lambda *argv: logging.log(99, ' '.join(map(str,argv)))

# sys.path.append('bm_to_vtk.pyz')
# platform_arch = '.cp%d%d-win_amd64' % (sys.hexversion >> 24, sys.hexversion >> 16 & 0xFF)
# pyd_path = os.environ['TEMP'] + "/pyz_" + platform_arch
# sys.path.insert(0, pyd_path)

import pyvista as pv
# try:
# except:
#   # we use this base class in enviroments that dont support VTK
#   class pv(object):
#     class StructuredGrid(object):
#       pass
#     class UniformGrid(object):
#       pass
#     def read(*argv):
#       pass
    
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
  #if pv is None: return
  if fp.lower().endswith('msh'):
    from _gui import leapfrog_load_mesh
    nodes, faces = leapfrog_load_mesh(fp)
    mesh = vtk_nf_to_mesh(nodes, faces)
  elif fp.lower().endswith('obj'):
    from _gui import wavefront_load_obj
    od = wavefront_load_obj(fp)
    mesh = pv.PolyData(np.asarray(od.get('v')), vtk_faces_to_cells(od.get('f')))
    if len(od.get('vt', [])):
      mesh.active_t_coords = np.array(od.get('vt'))
  elif fp.lower().endswith('00t'):
    from vulcan_save_tri import vulcan_load_tri
    nodes, faces, cv, cn = vulcan_load_tri(fp)
    mesh = vtk_nf_to_mesh(nodes, faces)
    # pyvista 0.26.1 and numpy on python 3.5 dont work due to np.flip
    #if cn == 'rgb':
    #  mesh.textures[0] = vtk_uint_to_texture(cv)

  elif re.search(r'gl(b|tf)$', fp, re.IGNORECASE):
    from pygltflib import GLTF2
    gltf = GLTF2.load(fp)
    mesh = gltf_to_vtk(gltf)
  elif re.search(r'vt(k|p|m)$', fp, re.IGNORECASE):
    mesh = pv.read(fp)
    import skimage.io
    for name in mesh.field_arrays:
      if len(name) == 1:
        v = mesh.field_arrays[name]
        mesh.textures[int(name)] = pv.Texture(np.reshape(v, (v.shape[0],-1,3)))

  else:
    from _gui import pd_load_dataframe
    df = pd_load_dataframe(fp)
    mesh = vtk_df_to_mesh(df)
  return mesh

def pv_save(meshes, fp, binary=True):
  ''' simple import safe pyvista writer '''
  if meshes is None: return
  if not hasattr(meshes, '__len__'):
    meshes = [meshes]

  if fp.lower().endswith('obj'):
    from _gui import wavefront_save_obj
    od = vtk_meshes_to_obj(meshes)
    wavefront_save_obj(fp, od)
  elif fp.lower().endswith('msh'):
    from _gui import leapfrog_save_mesh
    od = vtk_meshes_to_obj(meshes)
    leapfrog_save_mesh(od.get('v'), od.get('f'), fp)
  elif fp.lower().endswith('00t'):
    from vulcan_save_tri import vulcan_save_tri
    od = vtk_meshes_to_obj(meshes)
    vulcan_save_tri(od.get('v'), od.get('f'), fp)
  elif re.search(r'gl(b|tf)$', fp, re.IGNORECASE):
    gltf = vtk_to_gltf(meshes)
    gltf.save(fp)
  elif not re.search(r'vt(k|p|m)$', fp, re.IGNORECASE):
    df = pd.DataFrame()
    if not isinstance(meshes, list):
      meshes = [meshes]
    for mesh in meshes:
      df = df.append(vtk_mesh_to_df(mesh))
    from _gui import pd_save_dataframe
    pd_save_dataframe(df, fp)
  elif not isinstance(meshes, list):
    pv_save([meshes], fp, binary)
  elif len(meshes):
    for mesh in meshes:
      for k,v in mesh.textures.items():
        img = vtk_texture_to_array(v)
        mesh.field_data[str(k)] = np.reshape(img, (img.shape[0],-1))

    mesh = meshes[0]
    if len(meshes) > 1:
      mesh = pv.MultiBlock(meshes)
    
    mesh.save(fp, binary)

def vtk_cells_to_flat(cells):
  r = []
  p = 0
  n = None
  while p < len(cells):
    n = cells[p]
    r.extend(cells[p+1:p+1+n])
    p += n + 1
  return np.asarray(r), n

def vtk_flat_quads_to_triangles(quads, n = 4):
  f = []
  for i in range(0, len(quads), n):
    for j in range(0, n, 4):
      k = i + j
      #q = []
      f.extend(quads[k : k + 3])
      f.extend(quads[k + 2 : k + 4])
      f.append(quads[k])
      #print(len(q), i, ':', i + n // 2 + 1, i + n // 2, ':', i + n)
      #print(q)
      #faces.extend(q)
      # faces.extend(quads[i : i + n // 2 + 1])
      # faces.extend(quads[i + n // 2 : i + n])
      # faces.append(quads[i])
      # TODO: not working for 8 sided faces
      #print(i, i + n // 2 + 1, i + n // 2, i + n)
  return f

def vtk_cells_to_faces(cells):
  f, n = vtk_cells_to_flat(cells)
  #print(len(cells), len(f), n)
  if n is None:
    return f
  if n % 4 == 0:
    f = vtk_flat_quads_to_triangles(f, n)
  #print(len(f), (len(f) // 3) * 3)
  return np.reshape(f, (len(f) // 3, 3))

def vtk_flat_to_cells(flat, nodes = None):
  if nodes is None:
    nodes = pd.Series(np.arange(len(flat)), flat.index)
  n = 0
  cells = []
  for i in flat.index[::-1]:
    n += 1
    cells.insert(0, nodes[i])
    if flat[i] == 0:
      cells.insert(0, n)
      n = 0
  return np.array(cells)

def pd_detect_xyz(df, z = True):
  xyz = None
  dfcs = set(df.columns)
  for s in [['x','y','z'], ['midx','midy','midz'], ['mid_x','mid_y','mid_z'], ['xworld','yworld','zworld'], ['xcentre','ycentre','zcentre'], ['xc','yc','zc']]:
    if z == False:
      s.pop()
    for c in [str.lower, str.upper,str.capitalize]:
      cs = list(map(c, s))
      if dfcs.issuperset(cs):
        xyz = cs
        break
    else:
      continue
    # break also the outter loop if the inner loop broke
    break
  if xyz is None and z:
    return pd_detect_xyz(df, False)
  return xyz

def pd_detect_cell_size(df, xyz = None, xyzl = None):
  if xyz is None:
    xyz = pd_detect_xyz(df)
  if xyzl is None:
    xyzl = ['xlength', 'ylength', 'zlength']
  if set(xyzl).issubset(df):
    cell_size = df[xyzl].dropna().min().values
    if np.min(cell_size) <= 0:
      cell_size = None
    log("block length cell_size: ", cell_size)
  if cell_size is None:
    cell_size = np.full(len(xyz), np.nan)
    for i in range(len(xyz)):
      u = df[xyz[i]].unique()
      u = u[~np.isnan(u)]
      s = np.min(np.abs(np.subtract(u[1:], u[:-1])))
      if np.isnan(cell_size[i]) or s < cell_size[i]:
        cell_size[i] = s
    log("autodetect cell_size: ", cell_size)
  return cell_size

def getRectangleRotation(rect):
  r = 0
  #d = np.subtract(rect[0], rect[3])
  # x0y1 - x0y0
  d = np.subtract(rect[1], rect[0])
  print(d)
  if np.any(d):
    r = np.rad2deg(np.arctan(d[0]/d[1]))
  return r

def getRectangleSchema(rect, cell_size):
  dims = None
  origin = rect[0]
  bearing = 0
  o_n = None
  for i in range(len(rect)):
    if o_n is None or np.linalg.norm(rect[i]) < np.linalg.norm(rect[o_n]):
      o_n = i
  #o_s = 0
  dims = np.ones(2)
  # [n,n-1] , [n,n+1]
  d_s = []
  for k in [-1,+1]:
    d = 0
    s = 0
    n = o_n
    while d <= 0.001:
      n += k
      s = np.subtract(rect[o_n], rect[n])
      d = np.linalg.norm(s)
    d_s.append(d)
    print("i",i,"n",n,"o_n",o_n)
    # use last s
    o_s = s
  
  dims = np.ceil(np.asarray(d_s) / cell_size[:2])
  bearing = np.rad2deg(np.arctan(o_s[0]/o_s[1]))
  return rect[o_n], dims, bearing

def add_polygon_patch(coords, ax, fc = None):
  import matplotlib.patches as patches
  patch = patches.Polygon(np.array(coords.xy).T, fc=fc)
  ax.add_patch(patch)

def plt_polygon(p, ax = None):
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  if ax is None:
    ax = plt.gca()
  add_polygon_patch(p.exterior, ax)
  for interior in p.interiors:
    add_polygon_patch(interior, ax, 'w')
  ax.axis('equal')
  plt.show()

def pd_detect_rr(df, xyz = None):
  if xyz is None:
    xyz = pd_detect_xyz(df)
  from shapely.geometry import MultiPoint, Polygon

  pxyz = df.drop_duplicates(xyz[:2])
  pxyz = pxyz[xyz[:2]].values
  p2d = MultiPoint(pxyz)
  mrr = p2d.minimum_rotated_rectangle
  #return getRectangleRotation(mrr.exterior.coords)
  return mrr.exterior.coords

def vtk_faces_to_cells(faces):
  return np.hstack(np.concatenate((np.full((len(faces), 1), 3, dtype=np.int_), faces), 1))

def vtk_nf_to_mesh(nodes, faces):
  if len(nodes) == 0:
    return pv.PolyData()
  if len(faces) == 0:
    return pv.PolyData(np.array(nodes))
  #meshfaces = np.hstack(np.concatenate((np.full((len(faces), 1), 3, dtype=np.int_), faces), 1))
  return pv.PolyData(np.array(nodes), vtk_faces_to_cells(faces))

def vtk_df_to_mesh(df, xyz = None, dropna = False):
  # if pv is None: return
  if xyz is None:
    xyz = pd_detect_xyz(df)
  if xyz is None:
    log('geometry/xyz information not found')
    return None
  if len(xyz) == 2:
    xyz.append('z')
    if 'z' not in df:
      if '0' in df:
        # geotiff first/only spectral channel
        log('using first channel as Z value')
        df['z'] = df['0']
      else:
        log('using 0 as Z value')
        df['z'] = 0

  pdata = df[xyz]
  if dropna:
    pdata = pdata.dropna()
  # TODO: fix NaN without drop
  #print(pdata)

  if 'n' in df and df['n'].max() > 0:
    if 'node' in df:
      cells = vtk_flat_to_cells(df['n'], df['node'])
      nodes = df['node'].drop_duplicates().sort_values()
      pdata = pdata.loc[nodes.index]
    else:
      cells = vtk_flat_to_cells(df['n'])

    mesh = pv.PolyData(pdata.values.astype(np.float_), cells)
  else:
    mesh = pv.PolyData(pdata.values.astype(np.float_))

  for k,v in df.items():
    if k in xyz + ['w','t','n','closed','node']:
      continue
    try:
      mesh.point_arrays[k] = v[pdata.index]
    except:
      log("invalid column:", k)
  
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
  
  cv = [dict()] * grid.n_cells

  for i,row in df.iterrows():
    cell = grid.find_closest_cell(row[xyzc].values)
    if cell >= 0:
      cv[cell] = row[vl].to_dict()
  cvdf = pd.DataFrame.from_records(cv)
  for v in vl:
    grid.cell_data[v] = cvdf[v]

  return grid

def vtk_plot_meshes(meshes, point_labels=False, cmap = None, scalars = None):
  # plt.cm.terrain
  # plt.cm.plasma
  # plt.cm.gray
  # plt.cm.spectral
  # plt.cm.Paired
  # plt.cm.tab20
  # if pv is None: return
  p = pv.Plotter()
  
  if isinstance(cmap, str):
    import matplotlib.cm
    cmap = matplotlib.cm.get_cmap(cmap)

  c = 0
  if not hasattr(meshes, '__len__'):
    meshes = [meshes]
  for i in range(len(meshes)):
    mesh = meshes[i]
    if mesh is not None and mesh.n_points:
      # fix corner case of error when the plotter cant find a active scalar
      if mesh.active_scalars is None:
        for array_name in mesh.array_names:
          arr = mesh.get_array(array_name)
          if arr.dtype.num < 17:
            mesh.set_active_scalars(array_name)
            break
      color = None
      if cmap is not None:
        color = cmap(i/max(1,len(meshes)-1))
      if len(mesh.textures):
        p.add_mesh(mesh, color=None)
      elif mesh.GetDataObjectType() in [2,6]:
        p.add_mesh(mesh.slice_orthogonal(), opacity=0.5, scalars=scalars)
      else:
        p.add_mesh(mesh, opacity=0.5, color=color, scalars=scalars)

      if point_labels:
        p.add_point_labels(mesh.points, np.arange(mesh.n_points))
      c += 1
  if c:
    log("display", c, "meshes")
    p.add_axes()
    p.show()

def vtk_mesh_to_df(mesh, face_size = None, xyz = ['x','y','z'], n0 = 0):
  df = pd.DataFrame()
  if isinstance(mesh, list) or hasattr(mesh, 'n_blocks'):
    block_i = 0
    for block in mesh:
      bdf = vtk_mesh_to_df(block, face_size, xyz, n0)
      bdf['block'] = block_i
      df = df.append(bdf)
      block_i += 1
      n0 = df['node'].max() + 1
  else:
    #log("GetDataObjectType", mesh.GetDataObjectType())
    # VTK_STRUCTURED_POINTS = 1
    # VTK_STRUCTURED_GRID = 2
    # VTK_UNSTRUCTURED_GRID = 4
    # 6 = UniformGrid
    # VTK_UNIFORM_GRID = 10
    # 4 may or may not belong here
    points = None
    arr_node = None
    arr_n = None
    arr_data = None
    if mesh.GetDataObjectType() in [2,4,6]:
      points = mesh.cell_centers().points
      arr_n = np.zeros(mesh.n_cells, dtype=np.int_)
      arr_node = np.arange(mesh.n_cells, dtype=np.int_)
      arr_data = [pd.Series(mesh.get_array(name), name=name) for name in mesh.cell_arrays]
    else:
      arr_data = []
      # in some cases, n_faces may be > 0  but with a empty faces array
      if mesh.n_faces and len(mesh.faces):
        faces = None
        if face_size is None:
          faces, face_size = vtk_cells_to_flat(mesh.faces)
        else:
        #elif face_size < int(faces[0]):
          faces = vtk_cells_to_faces(mesh.faces)

        points = mesh.points.take(faces.flat, 0)
        #arr_n = np.tile(np.arange(face_size, dtype=np.int_), len(points) // face_size)
        # TODO: tile is rounding down, need better N generator
        arr_n = np.zeros(len(points), dtype=np.int_)
        for d in range(1, face_size):
          arr_n[d::face_size] += d
        arr_node = np.arange(mesh.n_faces, dtype=np.int_).take(faces.flat)
        #print(arr_node)
        for name in mesh.point_data:
          arr_data.append(pd.Series(mesh.get_array(name).take(faces.flat), name=name))
      else:
        points = mesh.points
        arr_n = np.zeros(mesh.n_points, dtype=np.int_)
        arr_node = np.arange(mesh.n_points, dtype=np.int_)
        arr_data = [pd.Series(mesh.get_array(name), name=name) for name in mesh.point_data if np.ndim(mesh.get_array(name)) == 1]

    df = pd.concat([pd.DataFrame(points, columns=xyz), pd.Series(arr_n, name='n'), pd.Series(np.add(arr_node, n0), name='node')] + arr_data, 1)
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
      arr_list = mesh.cell_data
      if preference == 'point':
        arr_list = mesh.point_data
      if preference == 'field':
        arr_list = mesh.field_data

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
  log("converting string arrays to integer index:")
  for name in mesh.cell_data:
    arr = mesh.cell_data[name]
    if arr.dtype.num >= 17:
      log(name,"(cell)",arr.dtype)
      mesh.cell_data[name] = pd.factorize(arr)[0]
  for name in mesh.point_data:
    arr = mesh.point_data[name]
    if arr.dtype.num >= 17:
      log(name,"(point)",arr.dtype)
      mesh.point_data[name] = pd.factorize(arr)[0]
  return mesh

class vtk_Voxel_(object):
  @classmethod
  def from_file_vtk(cls, *args):
    data = pv.read(args[0])
    log(data.GetDataObjectType())
    if data.GetDataObjectType() == 6:
      #self = cls(data.dimensions, data.spacing, data.origin)
      self = vtk_VoxelUG(data)
    elif data.GetDataObjectType() == 2:
      #origin = data.bounds[0::2]
      #spacing = np.divide(np.subtract(data.bounds[1::2], data.bounds[0::2]), data.extent[1::2])
      #self = cls(data.dimensions, spacing, origin)
      self = vtk_VoxelSG(data)
    else:
      self = data  
    #self.deep_copy(data)
    return self

  @classmethod
  def from_bmf(cls, bm, n_schema = None):
    if n_schema is None:
      n_schema = bm.model_n_schemas()-1
    else:
      n_schema = int(n_schema)

    size = np.resize(bm.model_schema_size(n_schema), 3)
    dims = bm.model_schema_dimensions(n_schema)
    o0 = bm.model_schema_extent(n_schema)
    origin = np.add(bm.model_origin(), o0[:3])

    self = cls(np.add(dims, 1, dtype = np.int_, casting = 'unsafe').tolist(), size, origin[:3])

    bearing, dip, plunge = bm.model_orientation()
    if abs(bearing - 90) > 0.01:
      # convert bearing to carthesian angle: A = -(B - 90)
      self = self.rotate_z_origin(bearing, origin)
    # store the raw rotation parameters as metadata
    self.field_arrays['orientation'] = [bearing, dip, plunge]
    self.field_arrays['origin'] = origin
    self.field_arrays['size'] = size
    
    return self

  def rotate_z_origin(self, bearing, point):
    r = - (bearing - 90)
    log("grid bearing: %.2f (%.f°)" % (bearing,r))
    self = vtk_VoxelSG(self.cast_to_structured_grid())
    # pyvista 26.0, last working for python 3.5
    # does not allow the rotation point
    if pv.__version__ == '0.26.1':
      self.translate(point * -1)
      self.rotate_z(r)
      self.translate(point)
    else:
      self.rotate_z(r, point)
    return self

  @classmethod
  def from_bb(cls, bb, cell_size = None, ndim = 3):
    dims = np.add(np.ceil(np.divide(np.subtract(bb[1], bb[0]), cell_size)), 2)

    if cell_size is None:
      cell_size = np.full(3, 10, dtype=np.int_)
    elif np.ndim(cell_size) == 0:
      cell_size = np.full(3, float(cell_size), dtype=np.int_)

    origin = np.subtract(bb[0], cell_size)
    if ndim == 2:
      dims[2] = 1
      origin[2] = 0
    return cls(dims.astype(np.int_).tolist(), cell_size, origin)
  
  @classmethod
  def from_bb_schema(cls, bb, schema, ndim = 3):
    bearing = 0
    s = schema.split(';')
    if len(s) > 1:
      bearing = float(s[1])
    cell_size = np.asarray(s[0].split(','), dtype=np.float_)
    if len(cell_size) < 3:
      cell_size = np.resize(cell_size, 3)
    
    if bearing != 0:
      mesh = pv.PolyData(bb)
      mesh.rotate_z(bearing, mesh.center)
    
      mesh_bb = np.transpose(np.reshape(mesh.bounds, (3,2)))

      bb[0] = np.min([bb[0], mesh_bb[0]], 0)
      bb[1] = np.max([bb[1], mesh_bb[1]], 0)

      self = cls.from_bb(bb, cell_size, ndim)
      self = vtk_VoxelSG(self.cast_to_structured_grid())
      self.rotate_z(bearing, mesh.center)
      return self
    else:
      return cls.from_bb(bb, cell_size, ndim)

  @classmethod
  def from_mesh(cls, mesh, cell_size = 10, ndim = 3):
    bb = np.transpose(np.reshape(mesh.bounds, (3,2)))
    return cls.from_bb(bb, cell_size, ndim)

  @classmethod
  def from_df(cls, df, cell_size = None, xyz = None, variables = None):
    if xyz is None:
      xyz = pd_detect_xyz(df)
    if cell_size is None:
      cell_size = pd_detect_cell_size(df, xyz)
    
    bb0 = df[xyz].min()
    bb1 = df[xyz].max()
    
    dims = np.add(np.ceil(np.divide(np.subtract(bb1, bb0), cell_size)), 2)

    origin = np.subtract(bb0.values, cell_size * 0.5)

    log("autodetect origin: %.2f,%.2f,%.2f" % tuple(origin))
    self = cls(dims.astype(np.int_).tolist(), cell_size, origin)
    if variables is None:
      variables = set(df.columns).difference(xyz)
    self.add_arrays_from_df(df, xyz, variables)
    return self

  def add_arrays_from_df(self, df, xyz, vl):
    if df.shape[0] == self.n_cells:
      # each cell matches with a df row
      for v in vl:
        self.cell_data[v] = df[v].values
    else:
      # find nearest cell using geometry
      # cache arrays. using directly from mesh.cell_data is bugged.
      # .to_numpy(np.float_)
      points = np.asfarray(df.filter(xyz))
      ci = self.find_closest_cell(points)
      
      for v in vl:
        # bool = 0
        # int32 = 7
        # int64 = 9
        data = np.ndarray(self.n_cells, dtype=df[v].dtype)
        if data.dtype.num <= 9:
          data[:] = -1
          np.put(data, ci, np.where(np.greater_equal(ci, 0), df[v].values, -1))
        else:
          data[:] = None
          np.put(data, ci, np.where(np.greater_equal(ci, 0), df[v].values, None))

        self.cell_data[v] = data
    
    return self

  @classmethod
  def from_rr(cls, df, cell_size = None, xyz = None, variables = None):
    ''' from automatic rotated rectangle '''
    if xyz is None:
      xyz = pd_detect_xyz(df)
    if cell_size is None:
      cell_size = pd_detect_cell_size(df, xyz)
    rr = pd_detect_rr(df)
    origin2d, dims2d, bearing = getRectangleSchema(rr, cell_size)
    origin = np.append(origin2d, df[xyz[2]].min())
    dims = np.append(dims2d, np.ceil(np.abs(np.subtract(df[xyz[2]].max(), df[xyz[2]].min()) / cell_size[2])))
    self = cls(dims.astype(np.int_).tolist(), cell_size, origin)
    print(bearing)
    bearing = 0
    if bearing:
      self = vtk_VoxelSG(self.cast_to_structured_grid())
      self.rotate_z(bearing, origin)
    if variables is None:
      variables = set(df.columns).difference(xyz)
    self.add_arrays_from_df(df, xyz, variables)
    return self

  @classmethod
  def from_file_path(cls, fp, rotate = False):
    ''' fire and forget parsing for multiple file types '''
    if not re.search(r'vt(k|p|m)$', fp, re.IGNORECASE):
      from _gui import pd_load_dataframe
      df = pd_load_dataframe(fp)
      if rotate:
        return cls.from_rr(df)
      else:
        return cls.from_df(df)
    else:
      return cls.from_file_vtk(fp)


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
      self.cell_data[name] = array.flat
    else:
      self.point_data[name] = array.flat

  def GetCellCenter(self, cellId):
    return vtk_Voxel_.sGetCellCenter(self, cellId)

  # DEPRECATED: use cell_centers().points
  @staticmethod
  def sGetCellCenter(self, cellId):
    cell = self.GetCell(cellId)
    bounds = np.reshape(cell.GetBounds(), (3,2))
    return bounds.mean(1)

  def get_elevation(self, mesh, fn = None):
    ''' 
    return the elevation of each cell relative to the given mesh 
    '''
    if fn is None:
      fn = np.mean
    cv = np.full(self.n_cells, np.nan)
    bounds = mesh.bounds
    cells = self.cell_centers().points
    for i in range(self.n_cells):
      #p0 = self.GetCellCenter(i)
      p0 = cells[i].copy()
      p1 = p0.copy()
      # create a line crossing the mesh bounding box in Z
      # TODO: use normals
      p0[2] = min(bounds[4], bounds[5]) - 1
      p1[2] = max(bounds[4], bounds[5]) + 1
      # check if the line hits the mesh anywhere
      ip, ic = mesh.ray_trace(p0, p1)
      if ip.size:
        # usualy a single point is returned for surfaces
        # but aggregate anyway to ensure a single Z value
        p = fn(ip, 0)
        cv[i] = p[2]

    return cv

  def cells_volume(self, v = None):
    ''' calculate a array with volume of each cell '''
    r = np.zeros(self.n_cells)
    for i in range(self.n_cells):
      b = self.GetCell(i).GetBounds()
      r[i] = abs(np.prod(np.subtract(b[1::2], b[0::2])))
    if v is not None:
      self.cell_data[v] = r
    return r

  def add_arrays_from_bmf(self, bm, condition = '', variables = None):
    if variables is None:
      variables = [_ for _ in bm.field_list() if not bm.field_predefined(_)] + ['xlength','ylength','zlength']
    elif isinstance(variables, str):
      variables = [variables]
    # its easy to make a UniformGrid, but we will need functions
    # only available to a StructuredGrid
    #grid = self.cast_to_structured_grid()
    #print("uniform", self.n_cells, "structured", grid.n_cells)

    cv = []
    #np.ndarray(grid.GetNumberOfCells(), dtype=[np.object_, np.float_, np.float_, np.float_, np.float_, np.int_][['name', 'integer', '***', 'float', 'double', 'bool'].index(bm.field_type(v))]) for v in variables]
    for i in range(len(variables)):
      j = ['name', 'integer', 'bool', '***', 'float', 'double'].index(bm.field_type(variables[i]))
      t = [np.object_, np.int_, np.int_, np.float_, np.float_, np.float_][j]
      n = ['', -1, -1, np.nan, np.nan, np.nan][j]
      cv.append(np.full(self.GetNumberOfCells(), n, dtype=t))

    bl = None
    if condition:
      block_select = bm_sanitize_condition(condition)
      bl = bm.get_matches(block_select)
    
    cells = self.cell_centers().points
    for cellId in range(self.GetNumberOfCells()):
      xyz = cells[cellId]
      if bm.find_world_xyz(*xyz):
        # point outside block model data region
        #self.BlankCell(cellId)
        pass
      elif bl is not None and bm.get_position() not in bl:
        #self.BlankCell(cellId)
        pass
      else:
        for i in range(len(variables)):
          if cv[i].dtype == np.object_:
            cv[i][cellId] = bm.get_string(variables[i])
          else:
            v = bm.get(variables[i])
            # vulcan None/NaN is -99
            if abs(v + 99) > 0.001:
              #v = np.nan
              cv[i][cellId] = v

    for i in range(len(variables)):
      self.cell_arrays[variables[i]] = cv[i]

    return self

  def ijk_array(self, array_name = None):
    shape = self.dimensions
    r = None
    if array_name is None:
      r = np.arange(self.n_cells).reshape(shape)
    else:
      if self.get_array_association(array_name) == pv.FieldAssociation.CELL:
        shape = np.subtract(self.dimensions, 1)
      r = self.get_array(array_name).reshape(shape)
    return r

  def heatmap2d(self, array_name, axis = 2, op = 'mean'):
    g3d = self.ijk_array(array_name)
    g2d = None
    if op == 'mean':
      # simple mean
      g2d = np.add.reduce(g3d, axis) / g3d.shape[axis]
    elif op == 'major':
      ft_a, ft_i = pd.factorize(g3d.flat)
      ft_a = np.reshape(ft_a, g3d.shape)
      fn = lambda _: pd.Series.value_counts(_).idxmax()
      g2d = np.apply_along_axis(fn, axis, ft_a)
    else:
      fn = eval('np.' + op)
      g2d = fn.reduce(g3d, axis)
    return g2d

class vtk_VoxelUG(pv.UniformGrid, vtk_Voxel_):
  pass

class vtk_VoxelSG(pv.StructuredGrid, vtk_Voxel_):
  pass

# backward compat
vtk_Voxel = vtk_VoxelUG

def vtk_texture_to_array(tex):
  ' drop in replacement for bugged to_array()'
  img = tex.to_image()
  sh = (img.dimensions[1], img.dimensions[0])
  if img.active_scalars.ndim > 1:
    sh = (img.dimensions[1], img.dimensions[0], tex.n_components)
  return img.active_scalars.reshape(sh)

def vtk_path_to_texture(fp):
  import skimage.io
  img = skimage.io.imread(fp)
  return pv.Texture(np.flip(img, 0))

def vtk_uint_to_texture(cv):
  rgb = [int(cv / 2 ** 16), int((cv % 2 ** 16) / 2**8), cv % 2**8]
  img = np.tile(np.multiply(rgb, 255), (8,8,1)).astype(dtype='uint8')
  return pv.Texture(img)

def vtk_rgb_to_texture(rgb):
  from matplotlib.colors import to_rgb
  img = np.tile(np.multiply(to_rgb(rgb), 255), (8,8,1)).astype(dtype='uint8')
  return pv.Texture(img)

def ireg_to_json(fp):
  import json
  s = open(fp).read()
  return json.loads(s.replace(' = u', ': NaN').replace('" = ', '": '))

def vtk_ireg_to_texture(mesh, fp):
  #from sklearn.metrics import pairwise_distances_argmin
  from sklearn.linear_model import LinearRegression
  ireg = ireg_to_json(fp)
  #df = pd.json_normalize(ireg['points'])
  image = np.array([_['image'] for _ in ireg['points']])
  world = np.array([_['world'] for _ in ireg['points']])
  reg = LinearRegression().fit(world, image)
  #print(reg.predict(world))
  #nni = pairwise_distances_argmin(mesh.points, world)
  mesh.active_t_coords = reg.predict(mesh.points)
  mesh.textures[0] = vtk_path_to_texture(ireg['properties']['image'])
  return mesh

def pretty_gltf(gltf):
  print(gltf.scenes)
  for _ in gltf.nodes:
    print(_)
  for _ in gltf.meshes:
    print(_)
  for _ in gltf.accessors:
    print(_)
  for _ in gltf.images:
    print(_)
  for _ in gltf.textures:
    print(_)
  for _ in gltf.materials:
    print(_)
  for _ in gltf.bufferViews:
    print(_)

def vtk_to_gltf(vtk_meshes, fp = None):
  import pygltflib
  from pygltflib.utils import ImageFormat
  import skimage.io
  import io
  buffer0 = io.BytesIO()
  nodes = []
  accessors = []
  bufferviews = []
  meshes = []
  textures = []
  images = []
  samplers = []
  materials = []
  for i_mesh in range(len(vtk_meshes)):
    mesh = vtk_meshes[i_mesh]
    faces = vtk_cells_to_faces(mesh.faces)
    tcoor = mesh.active_t_coords
    nodes.append(pygltflib.Node(mesh=i_mesh))
    # points
    position = len(accessors)
    view_blob = mesh.points.astype(np.float32).tobytes()
    bufferview = pygltflib.BufferView(buffer=0,byteOffset=buffer0.tell(),byteLength=len(view_blob),target=pygltflib.ARRAY_BUFFER)
    accessor = pygltflib.Accessor(bufferView=len(bufferviews),componentType=pygltflib.FLOAT,count=len(mesh.points),type=pygltflib.VEC3,max=mesh.points.max(axis=0).tolist(),min=mesh.points.min(axis=0).tolist())
    buffer0.write(view_blob)
    bufferviews.append(bufferview)
    accessors.append(accessor)
    # faces
    indices = len(accessors)
    view_blob = faces.astype(np.int_).tobytes()
    bufferview = pygltflib.BufferView(buffer=0,byteOffset=buffer0.tell(),byteLength=len(view_blob),target=pygltflib.ELEMENT_ARRAY_BUFFER)
    accessor = pygltflib.Accessor(bufferView=len(bufferviews),componentType=pygltflib.UNSIGNED_INT,count=faces.size,type=pygltflib.SCALAR,max=[],min=[])
    buffer0.write(view_blob)
    bufferviews.append(bufferview)
    accessors.append(accessor)
    # TEXCOORD
    texcoord = None
    if tcoor is not None:
      texcoord = len(accessors)
      view_blob = tcoor.astype(np.float32).tobytes()
      bufferview = pygltflib.BufferView(buffer=0,byteOffset=buffer0.tell(),byteLength=len(view_blob),target=pygltflib.ARRAY_BUFFER)
      accessor = pygltflib.Accessor(bufferView=len(bufferviews),componentType=pygltflib.FLOAT,count=len(tcoor),type=pygltflib.VEC2,max=[],min=[])
      buffer0.write(view_blob)
      bufferviews.append(bufferview)
      accessors.append(accessor)
    
    meshes.append(pygltflib.Mesh(primitives=[pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=position, TEXCOORD_0=texcoord), indices=indices, material=len(materials))]))

    for k,v in mesh.textures.items():
      img = vtk_texture_to_array(v)
      byteoffset = buffer0.tell()
      skimage.io.imsave(buffer0, img, format='png')
      # buffers chunks MUST be multiple of 4
      while buffer0.tell() % 4 > 0:
        buffer0.write(b'\0')
      materials.append(pygltflib.Material(doubleSided=True, alphaCutoff=None, pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(baseColorTexture=pygltflib.TextureInfo(index=len(textures), texCoord=0))))
      textures.append(pygltflib.Texture(source=int(k)))
      images.append(pygltflib.Image(mimeType=pygltflib.IMAGEPNG,bufferView=len(bufferviews)))
      bufferviews.append(pygltflib.BufferView(buffer=0,byteOffset=byteoffset,byteLength=buffer0.tell()-byteoffset))
    else:
      materials.append(pygltflib.Material(doubleSided=True, alphaCutoff=None))

  # 
  gltf = pygltflib.GLTF2(
      scene=0,
      scenes=[pygltflib.Scene(nodes=list(range(len(nodes))))],
      nodes=nodes,
      meshes=meshes,
      accessors=accessors,
      bufferViews=bufferviews,
      buffers=[
          pygltflib.Buffer(byteLength=buffer0.tell())
      ],
      images=images,
      samplers=samplers,
      textures=textures,
      materials=materials
  )

  gltf.set_binary_blob(buffer0.getbuffer())

  #pretty_gltf(gltf)

  if fp is not None:
    gltf.save(fp)

  return gltf

def dt2np(dt):
  import pygltflib
  d = {pygltflib.BYTE: 'byte', pygltflib.UNSIGNED_BYTE: 'ubyte', pygltflib.SHORT: 'short', pygltflib.UNSIGNED_SHORT: 'ushort', pygltflib.UNSIGNED_INT: 'uint32', pygltflib.FLOAT: 'float32'}
  return d.get(dt, 'float32')

def gltf_to_vtk(gltf):
  import skimage.io
  from io import BytesIO
  meshes = []
  # print(gltf.scenes)
  # print(gltf.nodes)
  # print(gltf.meshes)
  # print(gltf.accessors)
  # print(gltf.buffers)
  # print(gltf.bufferViews)
  # print(gltf.materials)
  # print(gltf.textures)
  # print(gltf.images)
  bb = gltf.binary_blob()
  for m in gltf.meshes:
    for p in m.primitives:
      ac = gltf.accessors[p.indices]
      bv = gltf.bufferViews[ac.bufferView]
      faces = np.frombuffer(bb[bv.byteOffset + ac.byteOffset : bv.byteOffset + bv.byteLength], dtype=dt2np(ac.componentType)).reshape((-1, 3))

      ac = gltf.accessors[p.attributes.POSITION]
      bv = gltf.bufferViews[ac.bufferView]

      nodes = np.frombuffer(bb[bv.byteOffset + ac.byteOffset : bv.byteOffset + bv.byteLength], dtype=dt2np(ac.componentType)).reshape((ac.count, -1))
      mesh = pv.PolyData(nodes, vtk_faces_to_cells(faces))
      if p.attributes.TEXCOORD_0 is not None:
        ac = gltf.accessors[p.attributes.TEXCOORD_0]
        bv = gltf.bufferViews[ac.bufferView]
        tc = np.frombuffer(bb[bv.byteOffset + ac.byteOffset : bv.byteOffset + bv.byteLength], dtype=dt2np(ac.componentType)).reshape((ac.count, -1))
        mesh.active_t_coords = tc

      t = gltf.materials[p.material]

      if t.pbrMetallicRoughness is not None:
        n = t.pbrMetallicRoughness.baseColorTexture.index
        bv = gltf.bufferViews[gltf.images[n].bufferView]
        raw = BytesIO(bb[bv.byteOffset : bv.byteOffset + bv.byteLength])
        img = skimage.io.imread(raw, format='png')
        mesh.textures[n] = pv.Texture(img)

      meshes.append(mesh)

  return meshes

def vtk_grid_to_mesh(grid, array_name = None, slices = 10):
  ''' grade shells - extract volumes of similar value as a mesh '''
  vtk_array_string_to_index(grid)
  if array_name is None:
    array_name = grid.active_scalars_name
  print("array_name", array_name)

  meshes = []
  if not array_name:
    grid = grid.elevation(set_active=True)
    array_name = grid.active_scalars_name

  dr = grid.get_data_range(array_name)
  #for r in range(int(dr[0]), int(dr[1] + 1), max(1, (dr[1] - dr[0]) // 99)):
  for r in np.linspace(dr[0], dr[1], slices):
    mesh = grid.threshold([r,r], array_name).extract_geometry()
    if mesh.n_cells:
      meshes.append(mesh)

  return meshes


def vtk_meshes_to_obj(meshes):
  ' convert a vtk mesh to a wavefront mesh dict '
  od = {"v": [], "f": [], "l": [], "vt": []}

  for mesh in meshes:
    c = None
    if hasattr(mesh, 'faces'):
      c = mesh.faces
    if hasattr(mesh, 'cells'):
      c = mesh.cells
    #for i in range(0, len(c), 9):
    #  print(c[i : i + 9])
    od['f'].extend(np.add(len(od['v']), vtk_cells_to_faces(c)))
    od['v'].extend(mesh.points)
    if mesh.t_coords is not None:
      od['vt'].extend(mesh.t_coords)

  return od

def vtk_meshes_bb(meshes, buffer = None):
  bounds0 = None
  bounds1 = None
  for mesh in meshes:
    if bounds0 is None:
      bounds0 = bounds1 = mesh.bounds
    else:
      bounds0 = np.min([bounds0, mesh.bounds], 0)
      bounds1 = np.max([bounds1, mesh.bounds], 0)
  bounds0 = bounds0[0::2]
  bounds1 = bounds1[1::2]
  if buffer:
    bounds0 = np.subtract(bounds0, buffer)
    bounds1 = np.add(bounds1, buffer)

  return np.stack([bounds0, bounds1])

class Raytracer(object):
  flag_cell = False
  def __init__(self, grid, flag_cell = False):
    self.flag_cell = flag_cell
    self.grid = grid
    n = 0
    #cv = np.full(grid.GetNumberOfPoints(), '', dtype=np.object_)
    if flag_cell:
      n = grid.GetNumberOfCells()
    else:
      n = grid.GetNumberOfPoints()
    #self.value = np.full(n, '', dtype=np.object_)
    self.value = np.full(n, None, dtype=np.object_)
  
  def _raytrace_cell(self, mesh, v):
    bounds = mesh.bounds
    cells = self.grid.cell_centers().points
    for i in range(len(cells)):
      p0 = cells[i].copy()
      p1 = p0.copy()
      # create a line crossing the mesh bounding box in Z
      p0[2] = min(bounds[4], bounds[5]) - 1
      p1[2] = max(bounds[4], bounds[5]) + 1
      # check if the line hits the mesh anywhere
      ip, ic = mesh.ray_trace(p0, p1)
      #print(p0, p1, ip.size, ic.size)
      if ic.size:
        self.value[i] = v

  def _raytrace_point(self, mesh, v):
    bounds = mesh.bounds
    for i in range(self.grid.n_points):
      p0 = self.grid.points[i].copy()
      p1 = p0.copy()
      # create a line crossing the mesh bounding box in Z
      p0[2] = min(bounds[4], bounds[5]) - 1
      p1[2] = max(bounds[4], bounds[5]) + 1
      # check if the line hits the mesh anywhere
      ip, ic = mesh.ray_trace(p0, p1)
      if ic.size:
        if v is None:
          # use Z
          self.value[i] = np.mean(ip, 0)[2]
        else:
          self.value[i] = v

  def raytrace(self, mesh, v = None):
    if self.flag_cell:
      return self._raytrace_cell(mesh, v)
    return self._raytrace_point(mesh, v)


if __name__=="__main__":
  pass
  #mesh = pv_read('std_voxel.csv')
  #print(mesh)
  df = pd.read_csv('std_voxel_r30.csv')
  mesh = vtk_Voxel.from_rr(df)
  print(mesh)
  # ,vtk_df_to_mesh(df)
  vtk_plot_meshes(mesh)

  