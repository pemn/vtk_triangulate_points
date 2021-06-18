# vtk_triangulate_points
Creates simplified surfaces from dense point clouds.  

## Objective
This script provides a open source and simple way to create simplified/lightweight triangulated surfaces (and solids) from dense point data.  
Input point data is usualy generated by traditional topography methods (toe and crest points). Those are best triangulated using Delaunay.  
But also can be generated by LIDAR (Laser), Drones (optical or radar) and even satellites (multiple spectral channels including IR). Those generate very dense point clouds which require a method other than Delaunay to create usable surfaces. In this cases "grid" is the recomended method.  
The grid method uses a voxel approach where the space is divided in cells and a single elevation value is estimated for each. If multiple points fall inside the cell, they will be aggregated into a single value. If no points are inside, the search region will be increased until it can find at least one point. This is different from traditional Machine Learning interpolation methods, which by default have to consider ALL sample points in EVERY grid cell, leading to inviable computation requirements.
## Installation
Download all files and run vtk_triangulate_points.py in your python enviroment of choice. The graphic interface should appear.  
Ex.:  
`python vtk_triangulate_points.py`  
Python 3.5+ required. Recomended: WinPython64-3.8.x.x (https://winpython.github.io/)  
The following modules are required:
- pandas
- sklearn
- pyvista  
  
Of those only pyvista is not commonly present is most python distros.
## Features
 - Multiple data file formats for input and output
 - Built in renderer to display results in a interative 3d window
 - Uses mature python libraries to do most tasks, nothing "new" to be mantained
 - Can be run either by the included graphic user interface or using command line arguments
 - Intermediary data is a pandas Dataframe so it can be easily extended/manipulated/exported/validated

## Engines
 - Grid: Useful for dense point clouds
 - Delaunay 2d: Best for exact toe/crest data or small point clouds
 - Delaunay 3d: same as 2d, but create a solid instead of surface
 - Outline: Create a simple visual bounding box. Useful to get a first look over the data extension when dealing with EXTREMELY large point clouds (Billions of points).

## Usage
The script accepts multiple data file formats common in the Mining, Cartography and Topography industries:
 - ASCII Csv
 - ESRI Shapefile
 - Autodesk DXF
 - Excel XLSX
 - VTK (*.vtk, *.vtm)
 - Las Topography (*.las)
 - Raster Geotiff (*.tif, *.tiff)

## Screenshots
### Graphic User Interface
![screenshot1](assets/screenshot1.png?raw=true)
### Example result using Grid method on bogota.tif
![screenshot5](assets/screenshot5.png?raw=true)
### Example result using Delaunay 2d on point_wall.csv
![screenshot2](assets/screenshot2.png?raw=true)
### Example Result using Delaunay 3d on point_wall.csv
![screenshot3](assets/screenshot3.png?raw=true)
### Example Result using Outline on point_wall.csv
![screenshot4](assets/screenshot4.png?raw=true)


