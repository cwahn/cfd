import os
import re
from operator import iconcat
from functools import partial, reduce
from itertools import filterfalse, tee
from attr import dataclass
from typing import *

import numpy as np
import pyvista as pv
import stl

Vector = Tuple[float, float, float]
TriFace = Tuple[Vector, Vector, Vector]
BoundingBox = Tuple[float, float, float, float, float, float]


@dataclass
class StlFace:
    normal: Vector
    points: TriFace


PATTERN = r"normal\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\souter\sloop\svertex\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\svertex\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\svertex\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"



def read_stl(path: str) -> pv.PolyData:
    reader = pv.get_reader(path)
    mesh = reader.read()
    return mesh


def save_stl(mesh: pv.PolyData, path: str):
    assert path.endswith(".stl"), "file name should be ends with .stl"

    temp_path = re.sub(".stl$", "_temp.stl", path)

    mesh.save(temp_path, binary=False)

    with open(temp_path, mode="r") as temp_f:
        with open(path, mode="w") as f:

            def is_solid_start(l):
                return (re.match("^solid.*$", l) != None)

            def rename(l):
                return "solid object\n" if is_solid_start(l) else l

            f.writelines(map(rename, temp_f))

    os.remove(temp_path)


def rotate(
        mesh: pv.PolyData,
        axis: Vector,
        theta_deg: float,
        point: Vector) -> pv.PolyData:
    rotated = mesh.rotate_vector(axis, theta_deg, point=point)
    return rotated


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
            ps[0][0], ps[0][1], ps[0][2]),
        "    vertex {} {} {}".format(
            ps[1][0], ps[1][1], ps[1][2]),
        "    vertex {} {} {}".format(
            ps[2][0], ps[2][1], ps[2][2]),
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
    liness = map(lambda i: get_face_stl(ps, i), idxs)
    lines = reduce(iconcat, liness, [])
    return lines


def save_with_boundary(
        path: str,
        mesh: pv.PolyData,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float):
    ps = get_points(x_min, x_max, y_min, y_max, z_min, z_max)
    faces_stl = get_faces_stl(ps)

    save_stl(mesh, path)

    with open(path, "a") as f:
        f.writelines(map(lambda l: l+"\n", faces_stl))


def prepare(
        in_path: str,
        out_path: str,
        axis: Vector,
        theta_degree: float,
        point: Vector,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float):
    mesh = read_stl(in_path)

    rotated = rotate(mesh, axis, theta_degree, point)

    save_with_boundary(out_path, rotated, x_min, x_max,
                       y_min, y_max, z_min, z_max)

    print(out_path)


def get_boolean_boxed(
        object: pv.PolyData,
        b_box: BoundingBox) -> pv.PolyData:
    
    box = pv.Box((b_box[0], b_box[1], b_box[2], b_box[3],
                  b_box[4], b_box[5])).triangulate()

    boxed = box.boolean_difference(object).triangulate()

    return boxed


def point_to_stlface_distance(point: Vector, face: StlFace) -> float:
    n = np.array(face.normal)
    p1 = np.array(face.points[0])
    d = - n.dot(p1)
    p = np.array(point)
    dist = abs(n.dot(p) + d) / np.linalg.norm(n)
    return dist


def is_stlsurfaces_parallel(s1: StlFace, s2: StlFace) -> bool:
    n1 = np.array(s1.normal)
    n2 = np.array(s2.normal)
    n1_hat = n1 / np.linalg.norm(n1)
    n2_hat = n2 / np.linalg.norm(n2)
    return abs(1 - abs(n1_hat.dot(n2_hat))) < 1e-9


def is_same_surf(s1: StlFace, s2: StlFace) -> bool:
    close_enough = point_to_stlface_distance(s1.points[0], s2) < 1e-9
    parallel = is_stlsurfaces_parallel(s1, s2)
    return close_enough and parallel


def match_to_stl_face(match: Iterable[str]) -> StlFace:
    ms = list(match)

    face = StlFace(
        (float(ms[0]), float(ms[1]), float(ms[2])),
        (
            (float(ms[3]), float(ms[4]), float(ms[5])),
            (float(ms[6]), float(ms[7]), float(ms[8])),
            (float(ms[9]), float(ms[10]), float(ms[11]))
        )
    )
    return face


def partition(pred, iterable):
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


def stl_face_to_lines(surf: StlFace) -> List[str]:
    n = surf.normal
    ps = surf.points
    lines = [
        " facet normal {} {} {}".format(n[0], n[1], n[2]),
        "  outer loop",
        "    vertex {} {} {}".format(
            ps[0][0], ps[0][1], ps[0][2]),
        "    vertex {} {} {}".format(
            ps[1][0], ps[1][1], ps[1][2]),
        "    vertex {} {} {}".format(
            ps[2][0], ps[2][1], ps[2][2]),
        "  endloop",
        " endfacet",
    ]
    return lines


def stl_faces_to_stl_solid_lines(faces: Iterable[StlFace], name: str) -> Iterable[str]:
    surf_lines = map(stl_face_to_lines, faces)
    lines = ["solid {}".format(
        name)] + reduce(iconcat, surf_lines, []) + ["endsolid"]
    return lines


def parse_stl_file(path: str) -> Iterable[StlFace]:
    with open(path, "r+") as f:
        stripted = map(lambda l: l.strip(), f)
        lines = map(lambda l: re.sub(" +", " ", l), stripted)
        joined_line = " ".join(lines)
        matches = re.findall(PATTERN, joined_line)
        stl_faces = map(match_to_stl_face, matches)
        return stl_faces


def save_and_rename_bounding_box(
        path: str,
        boxed_object: pv.PolyData,
        b_box: BoundingBox,
):
    temp_path = path.replace(".stl", "_.stl")

    save_stl(boxed_object, temp_path)

    stl_faces = parse_stl_file(temp_path)

    x_min_surf = StlFace(
        (-1, 0, 0), ((b_box[0], 0, 0), (b_box[0], 0, 0), (b_box[0], 0, 0)))
    x_max_surf = StlFace(
        (1, 0, 0), ((b_box[1], 0, 0), (b_box[1], 0, 0), (b_box[1], 0, 0)))
    y_min_surf = StlFace(
        (0, -1, 0), ((0, b_box[2], 0), (0, b_box[2], 0), (0, b_box[2], 0)))
    y_max_surf = StlFace(
        (0, 1, 0), ((0, b_box[3], 0), (0, b_box[3], 0), (0, b_box[3], 0)))
    z_min_surf = StlFace(
        (0, 0, -1), ((0, 0, b_box[4]), (0, 0, b_box[4]), (0, 0, b_box[4])))
    z_max_surf = StlFace(
        (0, 0, 1), ((0, 0, b_box[5]), (0, 0, b_box[5]), (0, 0, b_box[5])))

    b_faces = [x_min_surf, x_max_surf, y_min_surf,
               y_max_surf, z_min_surf, z_max_surf]

    predicates = list(map(lambda x: partial(is_same_surf, s2=x), b_faces))

    # Monad is here
    left, x_mins = partition(predicates[0], stl_faces)
    left, x_maxs = partition(predicates[1], left)
    left, y_mins = partition(predicates[2], left)
    left, y_maxs = partition(predicates[3], left)
    left, z_mins = partition(predicates[4], left)
    left, z_maxs = partition(predicates[5], left)
    objects = left

    solids = [x_mins, x_maxs, y_mins, y_maxs, z_mins, z_maxs, objects]
    names = ["xMin", "xMax", "yMin", "yMax", "zMin", "zMax", "object"]

    solid_liness = map(lambda p: stl_faces_to_stl_solid_lines(
        p[0], p[1]), zip(solids, names))

    outputs = list(reduce(iconcat, solid_liness, []))

    with open(path, "w") as o:
        o.writelines(list(map(lambda x: x+"\n", outputs)))


def prepare2(
        in_path: str,
        out_path: str,
        axis: Vector,
        theta_deg: float,
        point: Vector,
        b_box: BoundingBox):
    object = read_stl(in_path)
    rotated = rotate(object, axis, theta_deg, point)
    boxed = get_boolean_boxed(rotated, b_box)
    save_and_rename_bounding_box(out_path, boxed, b_box)
