from functools import partial
from math import pi
from re import sub
import re
from typing import Any, Iterable, List, Tuple
from matplotlib.collections import TriMesh
from more_itertools import flatten, split_after
import trimesh
from trimesh import Trimesh, transformations, boolean


Vector = Tuple[float, float, float]
TriFace = Tuple[Vector, Vector, Vector]
BoundingBox = Tuple[float, float, float, float, float, float]


def read_mesh(path: str) -> Trimesh:
    return trimesh.load(path, force="mesh")

def get_b_box(b_box: BoundingBox) -> Trimesh:
    bounds = ((b_box[0], b_box[2], b_box[4]), (b_box[1], b_box[3], b_box[5]))
    return trimesh.creation.box(bounds=bounds)

def rotate_ccw(object:Trimesh, angle_deg: float) -> Trimesh:
    angle = angle_deg * pi/180. 
    direction = [0, 1, 0]
    center = [0, 0, 0]

    transform_mtx = transformations.rotation_matrix(angle, direction, center)
    return object.apply_transform(transform_mtx)

def boolean_diff(lhs: Trimesh, rhs: Trimesh) -> Trimesh:
    return boolean.difference([lhs, rhs], engine="blender")

def boolean_union(lhs: Trimesh, rhs: Trimesh) -> Trimesh:
    return boolean.union([lhs, rhs], engine="blender")

def separate(object: Trimesh, b_box: BoundingBox) -> Iterable[Trimesh]:
    eps = 1e-9

    b_box_ = [
        b_box[0] + eps,
        b_box[1] - eps,
        b_box[2] + eps,
        b_box[3] - eps,
        b_box[4] + eps,
        b_box[5] - eps,
    ]

    vs = object.vertices

    x_min_surf = []
    x_max_surf = []
    y_min_surf = []
    y_max_surf = []
    z_min_surf = []
    z_max_surf = []
    obj = []

    for f in object.faces:
        # f: (idx, idx, idx)
        vert0 = object.vertices[f[0]] # (n,3) float
        vert1 = object.vertices[f[1]]
        vert2 = object.vertices[f[2]]

        if vert0[0] < b_box_[0] and vert1[0] < b_box_[0] and vert2[0] < b_box_[0]:
            x_min_surf.append(f)
            continue

        if vert0[0] > b_box_[1] and vert1[0] > b_box_[1] and vert2[0] > b_box_[1]:
            x_max_surf.append(f)
            continue

        if vert0[1] < b_box_[2] and vert1[1] < b_box_[2] and vert2[1] < b_box_[2]:
            y_min_surf.append(f)
            continue

        if vert0[1] > b_box_[3] and vert1[1] > b_box_[3] and vert2[1] > b_box_[3]:
            y_max_surf.append(f)
            continue

        if vert0[2] < b_box_[4] and vert1[2] < b_box_[4] and vert2[2] < b_box_[4]:
            z_min_surf.append(f)
            continue

        if vert0[2] > b_box_[5] and vert1[2] > b_box_[5] and vert2[2] > b_box_[5]:
            z_max_surf.append(f)
            continue 

        obj.append(f)

    facess = [x_min_surf, x_max_surf, y_min_surf,
               y_max_surf, z_min_surf, z_max_surf, obj]
    
    objects = map(lambda fs: Trimesh(vs, faces=fs), facess)

    return objects

    

def add_name(name: str, content: str) -> str:
    named = re.sub("^solid\s\n", f"solid {name} \n", content)
    # print(named + " \n")
    return named + " \n"

def save_named_mesh(object: Trimesh, b_box: BoundingBox, path: str):
    objects_faces = separate(object, b_box)
    strs: Iterable[str] = map(lambda o: str(o.export(file_type="stl_ascii")), objects_faces)
    names = ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax", "object"]
    name_and_strs = zip(names, strs)
    named_strs = map(lambda t: add_name(t[0], t[1]), name_and_strs)
    # lines = flatten(named_liness)

    with open(path, "w") as f:
        f.writelines(named_strs)


def prepare_mesh(
        in_path: str,
        out_path: str,
        background_mesh: TriMesh,
        building_rotation_deg: float,
        wind_direction_deg: float,
        bounding_box: BoundingBox,
):
    building = read_mesh(in_path)

    rotated_building = rotate_ccw(building, building_rotation_deg)

    joined: Trimesh = boolean_union(background_mesh, rotated_building)

    rotated_obj = rotate_ccw(joined, 90 - wind_direction_deg)

    b_box = get_b_box(bounding_box)

    result = boolean_diff(b_box, rotated_obj)

    save_named_mesh(result, bounding_box, out_path)

