from typing import Tuple
from more_itertools import last
from numpy import arange, concatenate, linspace, mean, meshgrid, ndarray, ravel, sqrt
from pyvista import MultiBlock, OpenFOAMReader, StructuredGrid, UnstructuredGrid
from sklearn.preprocessing import normalize

import numpy as np

from openfoam.mesh import BoundingBox

def get_last_time(reader: OpenFOAMReader) -> float:
    times = reader.time_values
    last_time = last(times)
    return last_time


def read_case_last(path: str) -> MultiBlock:
    reader = OpenFOAMReader(path)
    last_time = get_last_time(reader)
    reader.set_active_time_value(last_time)
    meshes: MultiBlock = reader.read()
    return meshes

def get_bbox(mesh: UnstructuredGrid) -> BoundingBox:
    attrs = mesh._get_attrs()
    
    xs = attrs[2][1]
    ys = attrs[3][1]
    zs = attrs[4][1]

    return (xs[0], xs[1], ys[0], ys[1], zs[0], zs[1])

def grid_from_bbox_odd(b_box: BoundingBox, grid_size: float) -> StructuredGrid:
    xn = arange(-grid_size, b_box[0], -grid_size)
    xp = arange(0, b_box[1], grid_size)
    x = concatenate([xn, xp])

    yn = arange(-grid_size, b_box[2], -grid_size)
    yp = arange(0, b_box[3], grid_size)
    y = concatenate([yn, yp])

    zn = arange(-grid_size, b_box[4], -grid_size)
    zp = arange(0, b_box[5], grid_size)
    z = concatenate([zn, zp])

    xs, ys, zs = meshgrid(x, y, z)

    return StructuredGrid(xs, ys, zs)

def grid_from_bbox_even(b_box: BoundingBox, grid_size: float) -> StructuredGrid:
    xn = list(reversed(arange(-grid_size/2, b_box[0], -grid_size)))
    xp = arange(grid_size/2, b_box[1], grid_size)
    x = concatenate([xn, xp])

    yn = list(reversed(arange(-grid_size/2, b_box[2], -grid_size)))
    yp = arange(grid_size/2, b_box[3], grid_size)
    y = concatenate([yn, yp])

    zn = list(reversed(arange(-grid_size/2, b_box[4], -grid_size)))
    zp = arange(grid_size/2, b_box[5], grid_size)
    z = concatenate([zn, zp])
    # print("x: ", x)

    xs, ys, zs = meshgrid(x, y, z, indexing='ij')
    # print("xs: ", xs)

    return StructuredGrid(xs, ys, zs)


def spatial_sample_case_odd(path: str, b_box: BoundingBox, grid_size: float) -> Tuple[StructuredGrid, Tuple]:
    meshes = read_case_last(path)
    internal_mesh = meshes["internalMesh"]
    
    # Can't trust this. Use orinal b_box
    # b_box = get_bbox(internal_mesh) 
    grid = grid_from_bbox_odd(b_box, grid_size)
    sampled = grid.interpolate(
        internal_mesh, 
        radius = grid_size, 
        strategy="closest_point")

    return (sampled, grid.dimensions)

def spatial_sample_case_even(path: str, b_box: BoundingBox, grid_size: float) -> Tuple[StructuredGrid, Tuple]:
    meshes = read_case_last(path)
    internal_mesh = meshes["internalMesh"]
    
    grid = grid_from_bbox_even(b_box, grid_size)
    
    sampled = grid.interpolate(
        internal_mesh, 
        radius = grid_size, 
        strategy="closest_point")

    return (sampled, grid.dimensions)

def reshape_sampled(sampled: StructuredGrid, grid_shape: Tuple[float, float, float]) -> np.array:
    u = sampled["U"]
    p = sampled["p"]
    k = sampled["k"]

    fs = [u, p, k]

    reshaped = map(lambda f: f.reshape((grid_shape[0], grid_shape[1], grid_shape[2], -1)), fs)

    transposed = map(lambda f: np.transpose(f), reshaped)

    res = np.concatenate(list(transposed), axis=0)

    return res

def normalize_upk(sampled: StructuredGrid) -> Tuple[ndarray, ndarray, ndarray]:
    u = sampled["U"].reshape(-1,1)
    p = sampled["p"].reshape(-1,1)
    k = sampled["k"].reshape(-1,1)

    return tuple(map(normalize, [u, p, k]))

def ndarray_rms(lhs: ndarray, rhs: ndarray) -> float:
    return sqrt(mean(((lhs - rhs)**2)))
    
def rms_of_sampled(lhs: StructuredGrid, rhs: StructuredGrid) -> float:
    upk_l = normalize_upk(lhs)
    upk_r = normalize_upk(rhs)

    lss = list(map(ravel, upk_l))
    rss = list(map(ravel, upk_r))

    ls = concatenate(lss)
    rs = concatenate(rss)

    return ndarray_rms(ls, rs)

def post_process(foam_path: str, b_box: BoundingBox, grid_size: int) -> np.array:
    sampled, grid_size = spatial_sample_case_even(foam_path, b_box, grid_size)
    res = reshape_sampled(sampled, grid_size)
    return res