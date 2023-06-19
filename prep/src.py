from cProfile import label
from functools import partial, reduce
from operator import iconcat
import os
import re
from typing import *
from numpy import indices, save

import pyvista as pv

Vector = Tuple[float, float, float]
TriFace = Tuple[Vector, Vector, Vector]


def read_stl(file_name: str) -> pv.PolyData:
    reader = pv.get_reader(file_name)
    mesh = reader.read()
    return mesh


def save_stl(mesh: pv.PolyData, file_name: str):
    assert file_name.endswith(".stl"), "file name should be ends with .stl"

    temp_file_name = re.sub(".stl$", "_temp.stl", file_name)

    mesh.save(temp_file_name, binary=False)

    with open(temp_file_name, mode="r") as temp_file:
        with open(file_name, mode="w") as file:

            def is_solid_start(l):
                return (re.match("^solid.*$", l) != None)

            def rename(l):
                return "solid object\n" if is_solid_start(l) else l

            file.writelines(list(map(rename, temp_file)))
    os.remove(temp_file_name)


def rotate(mesh: pv.PolyData, axis: Vector, theta_degree: float, point: Vector) -> pv.PolyData:
    rotated = mesh.rotate_vector(axis, theta_degree, point=point)
    return rotated


# def get_surface_raw(seed: Tuple[int, float], ps: List[Vector]) -> pv.PolyData:
#     i = seed[0]
#     val = seed[1]

#     filtered = filter(lambda e: e[1][i] == val, enumerate(ps))
#     print(filtered)
#     vertices = list(map(lambda e: e[1], filtered))
#     indices = list(map(lambda e: e[0], filtered))
#     faces = [4]+indices
#     print(faces)

#     mesh = pv.PolyData(vertices, faces)
#     return mesh


# def get_bounding_surfaces(x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float) -> List[pv.PolyData]:
#     ps = get_points(x_min, x_max, y_min, y_max, z_min, z_max)

#     get_surface = partial(get_surface_raw, ps=ps)

#     seeds = [(0, x_min), (0, x_max), (1, y_min),
#              (1, y_max), (2, z_min), (2, z_max)]

#     surfaces = list(map(get_surface, seeds))

#     return surfaces


def get_points(x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float) -> List[Vector]:
    ps = [
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_max, y_max, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_max),
        (x_max, y_min, z_max),
        (x_max, y_max, z_max),
        (x_min, y_max, z_max),
    ]
    return ps


def get_name_normal_idxs(boundary_idx: int) -> Tuple[str, Vector, List[List[int]]]:
    if boundary_idx == 0:
        normal = (-1, 0, 0)
        idxss = [
            [0, 4, 7],
            [0, 3, 7]
        ]
        return ("xMin", normal, idxss)
    elif boundary_idx == 1:
        normal = (1, 0, 0)
        idxss = [
            [1, 2, 6],
            [1, 6, 5]
        ]
        return ("xMax", normal, idxss)
    elif boundary_idx == 2:
        normal = (0, -1, 0)
        idxss = [
            [0, 1, 5],
            [0, 5, 4]
        ]
        return ("yMin", normal, idxss)
    elif boundary_idx == 3:
        normal = (0, 1, 0)
        idxss = [
            [2, 3, 7],
            [2, 7, 6]
        ]
        return ("yMax", normal, idxss)
    elif boundary_idx == 4:
        normal = (0, 0, -1)
        idxss = [
            [0, 3, 2],
            [0, 2, 1]
        ]
        return ("zMin", normal, idxss)
    elif boundary_idx == 5:
        normal = (0, 0, 1)
        idxss = [
            [4, 5, 6],
            [4, 6, 7]
        ]
        return ("zMax", normal, idxss)


def get_face_stl(ps: List[Vector], boundary_idx: int) -> List[str]:
    (name, normal, idxss) = get_name_normal_idxs(boundary_idx)

    def idxs_to_triface(idxs): return (ps[idxs[0]], ps[idxs[1]], ps[idxs[2]])
    faces = list(map(idxs_to_triface, idxss))

    lines = [
        "solid {}".format(name),
        " facet normal {} {} {}".format(normal[0], normal[1], normal[2]),
        "  outer loop",
        "    vertex {} {} {}".format(
            faces[0][0][0], faces[0][0][1], faces[0][0][2]),
        "    vertex {} {} {}".format(
            faces[0][1][0], faces[0][1][1], faces[0][1][2]),
        "    vertex {} {} {}".format(
            faces[0][2][0], faces[0][2][1], faces[0][2][2]),
        "  endloop",
        " endfacet",
        " facet normal {} {} {}".format(normal[0], normal[1], normal[2]),
        "  outer loop",
        "    vertex {} {} {}".format(
            faces[1][0][0], faces[1][0][1], faces[1][0][2]),
        "    vertex {} {} {}".format(
            faces[1][1][0], faces[1][1][1], faces[1][1][2]),
        "    vertex {} {} {}".format(
            faces[1][2][0], faces[1][2][1], faces[1][2][2]),
        "  endloop",
        " endfacet",
        "endsolid",
    ]
    return lines


def get_faces_stl(ps: List[Vector]) -> List[str]:
    idxs = [0, 1, 2, 3, 4, 5]
    liness = list(map(lambda i: get_face_stl(ps, i), idxs))
    lines = list(reduce(iconcat, liness, []))
    return lines


def save_with_boundary(
        file_name: str,
        mesh: pv.PolyData,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float):
    ps = get_points(x_min, x_max, y_min, y_max, z_min, z_max)
    faces_stl = get_faces_stl(ps)

    save_stl(mesh, file_name)

    with open(file_name, "a") as f:
        f.writelines(list(map(lambda l: l+"\n", faces_stl)))


def prepare(
        in_file_name: str,
        out_file_name: str,
        axis: Vector,
        theta_degree: float,
        point: Vector,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float):
    mesh = read_stl(in_file_name)

    rotated = rotate(mesh, axis, theta_degree, point)

    save_with_boundary(out_file_name, rotated, x_min, x_max,
                       y_min, y_max, z_min, z_max)
    
    print(out_file_name)
