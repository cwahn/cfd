
from functools import partial
from genericpath import isdir
from os import listdir, mkdir, path
import pickle
import random
from re import findall
from shutil import copy2
from time import sleep
from typing import List, Tuple
from openfoam.core import Config, configure, run_all, run_case, run_case_no_meshing, run_no_meshing, set_case
from openfoam.mesh import BoundingBox
from openfoam.post import spatial_sample_case_even
import numpy as np
from openfoam.prep import StlFace, get_boolean_boxed, is_same_surf, parse_stl_file, partition, save_stl
from prep.src import read_stl

# MAX_CELL_SIZE = 97.59472956
# BOUNDARY_CELL_SIZE = 52.8544593
# LOCAL_REF_CELL_SIZE = 10.49705968
# RESIDUAL_P = 0.544521855
# RESIDUAL_OTHERS = 0.042843653

MAX_CELL_SIZE = 100
BOUNDARY_CELL_SIZE = 52
LOCAL_REF_CELL_SIZE = 10.5
RESIDUAL_P = 0.5
RESIDUAL_OTHERS = 0.04

B_BOX = [-500, 500, 1e-2, 500, -500, 500]

LEN_X = (B_BOX[1]-B_BOX[0])
LEN_Y = (B_BOX[3]-B_BOX[2])
LEN_Z = (B_BOX[5]-B_BOX[4])

GRID_SIZE = 3.9
W, H, D = (256, 128, 256)
RATIO = [LEN_X/W, LEN_Y/H, LEN_Z/D]

def parse_shape_file_name(shape_file_name: str) -> tuple[int, float, float]:
    shape_num, rel_rot, wind_rot = findall("^(.*)-.*-(.*)-(.*).stl", shape_file_name)[0]
    return (int(shape_num), float(rel_rot), float(wind_rot))

def get_cfd_arg_batches(c: Config) -> List[Tuple[str, List[float]]]:
    local_shape_dir_path = path.join(c.local_case_dir_path, "constant/triSurface/")

    shape_file_names = listdir(local_shape_dir_path)

    case_rel_shape_paths = list(map(lambda x: path.join("constant/triSurface/", x), shape_file_names))

    # print(shape_file_names[0])
    shape_nums= list(map(lambda f: parse_shape_file_name(f)[0], shape_file_names)) 

    path_shape_num = zip(case_rel_shape_paths, shape_nums)

    random.seed(1)

    def rands(i: int) -> List[float]:
        return list(map(lambda x: random.uniform(0, 30), [0]*i))
        
    path_with_speeds_nums = list(map(lambda p: (p[0], rands(10), p[1]), path_shape_num))
    
    return path_with_speeds_nums

def get_entries(
        c: Config,
        case_rel_shape_path: str,
        wind_speed: float,
        # MAX_CELL_SIZE: float,
        # BOUNDARY_CELL_SIZE: float,
        # LOCAL_REF_CELL_SIZE: float,
        # RESIDUAL_P: float,
        # RESIDUAL_OTHERS: float
        ) -> List:

    entries = [
            ("0/U", ["Uinlet"], f"({wind_speed} 0 0)"),
            ("system/meshDict", ["surfaceFile"], case_rel_shape_path),
            ("system/meshDict", ["maxCellSize"], MAX_CELL_SIZE),
            ("system/meshDict", ["boundaryCellSize"], BOUNDARY_CELL_SIZE),
            ("system/meshDict", ["localRefinement", "object.*", "cellSize"], LOCAL_REF_CELL_SIZE),
            ("system/meshDict", ["boundaryLayers", "patchBoundaryLayers", "object.*", "nLayers"], 5),
            ("system/meshDict", ["boundaryLayers", "patchBoundaryLayers", "object.*", "thicknessRatio"], 1.2),
            ("system/fvSolution", ["SIMPLE", "residualControl", "p"], RESIDUAL_P),
            ("system/fvSolution", ["SIMPLE", "residualControl", "U"], RESIDUAL_OTHERS),
            ("system/fvSolution", ["SIMPLE", "residualControl", "k"], RESIDUAL_OTHERS),
            ("system/fvSolution", ["SIMPLE", "residualControl", "omega"], RESIDUAL_OTHERS),
            ("system/fvSolution", ["SIMPLE", "residualControl", "epsilon"], RESIDUAL_OTHERS),
            ("system/decomposeParDict", ["numberOfSubdomains"], c.num_proc),
        ]
    return entries

def dot2(x):
    return np.dot(x, x)

def sdf_triangle(p, a, b, c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    ba = b - a 
    pa = p - a
    cb = c - b
    pb = p - b
    ac = a - c
    pc = p - c
    nor = np.cross(ba, ac)

    s = np.sign(np.dot(np.cross(ba,nor),pa)) + np.sign(np.dot(np.cross(cb,nor),pb)) + np.sign(np.dot(np.cross(ac,nor),pc)) < 2.0
    
    return np.sqrt(
        min( 
        min( 
        dot2(ba*np.clip(np.dot(ba,pa)/dot2(ba),0.0,1.0)-pa), 
        dot2(cb*np.clip(np.dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),  dot2(ac*np.clip(np.dot(ac,pc)/dot2(ac),0.0,1.0)-pc) ) if s else np.dot(nor,pa)*np.dot(nor,pa)/dot2(nor) 
    )

def si(p, a, b, c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    ba = b - a 
    ac = a - c
    nor = np.cross(ba, ac)
    return np.sign(np.dot(p-a, nor))


# save_stl 저장 불필요 
def rename_bounding_box(
        path: str,
        b_box: BoundingBox,
):
    temp_path = path.replace(".stl", "_.stl")

    stl_faces = parse_stl_file(path)

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

    return  list(y_mins) + list(objects)



def process_x_once(c: Config, rel_case_shape_path: str, local_output_dir_path: str) :
    
    abs_shape_path = path.join(c.local_case_dir_path, rel_case_shape_path)
    copy2(abs_shape_path, local_output_dir_path)
    
    
def process_y(result_path: str, b_box: BoundingBox, grid_size: float) -> np.array:
    foam_path = path.join(result_path, "open.foam")
    print(foam_path)
    sampled = spatial_sample_case_even(foam_path, b_box, grid_size)
    u = sampled["U"]
    p = sampled["p"].reshape(-1, 1)
    k = sampled["k"].reshape(-1, 1)

    res = np.hstack([u, p, k]).reshape(256,128,256,5)

    print(res.shape)
    return res

def write_data(x, y, write_path: str):
    with open(write_path,"wb") as fw:
        pickle.dump((x, y), fw)

def get_file_name(file_path: str) -> str:
    return path.basename(file_path).split('/')[-1]

def prepare_output_dir(c: Config, i: int) -> str:
    write_dir_path = path.join(c.local_case_dir_path, f"output{i}/")

    if not isdir(write_dir_path):
        mkdir(write_dir_path)

    return write_dir_path


def gen_data(c: Config):  

    path_with_speeds_nums = get_cfd_arg_batches(c)
    
    for i, case_rel_path_speeds in enumerate(path_with_speeds_nums):

        print("Processing shape ", i)

        case_rel_shape_path = case_rel_path_speeds[0]

        local_output_dir_path = prepare_output_dir(c, case_rel_path_speeds[2])   

        process_x_once(c, case_rel_shape_path, local_output_dir_path)

        for j, speed in enumerate(case_rel_path_speeds[1]):

            entries = get_entries(c, case_rel_shape_path, speed)

            if j == 0:
                run_case(c, entries, 120)
            else:
                run_case_no_meshing(c, entries, 120)

            x = speed
            y = process_y(c.local_case_dir_path, B_BOX, GRID_SIZE)

            write_path = path.join(local_output_dir_path, f"{speed}-" + get_file_name(case_rel_shape_path))

            write_data(x, y, write_path)

            
