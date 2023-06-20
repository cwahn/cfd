
from functools import partial
from os import listdir, mkdir, path
import pickle
import random
from re import findall
from typing import List, Tuple
from openfoam.core import Config, configure, run_all, run_case, run_case_no_meshing, run_no_meshing, set_case
from openfoam.mesh import BoundingBox
from openfoam.post import spatial_sample_case_even
import numpy as np
from openfoam.prep import StlFace, get_boolean_boxed, is_same_surf, parse_stl_file, partition, save_stl
from prep.src import read_stl

MAX_CELL_SIZE = 97.59472956
BOUNDARY_CELL_SIZE = 52.8544593
LOCAL_REF_CELL_SIZE = 10.49705968
RESIDUAL_P = 0.544521855
RESIDUAL_OTHERS = 0.042843653

B_BOX = [-500, 500, 1e-2, 500, -500, 500]

LEN_X = (B_BOX[1]-B_BOX[0])
LEN_Y = (B_BOX[3]-B_BOX[2])
LEN_Z = (B_BOX[5]-B_BOX[4])

GRID_SIZE = 3.9
W, H, D = (256, 128, 256)
RATIO = [LEN_X/W, LEN_Y/H, LEN_Z/D]

# def parse_wind_dir(path: str) -> float:
#         wind_dir = findall("-([0-9]*\.?[0-9]*).stl")[0]
#         return wind_dir

def get_cfd_arg_batches(c: Config) -> List[Tuple[str, List[float]]]:
    local_shape_dir_path = path.join(c.local_volum_path, "constant/triSurface/")
    # container_shape_dir_path = path.join(c.container_mount_path, "constant/triSurface/")
    shape_paths = listdir(local_shape_dir_path)

    random.seed(1)

    def rands(i: int) -> List[float]:
        return list(map(lambda x: random.uniform(0, 30), [0]*i))
        
    # wind_dirs = map(parse_wind_dir, shape_paths)
    path_with_speedss = list(map(lambda p: (p, rands(10)), shape_paths))
    return path_with_speedss

def get_entries(
        c: Config,
        path: str,
        wind_speed: float,
        # MAX_CELL_SIZE: float,
        # BOUNDARY_CELL_SIZE: float,
        # LOCAL_REF_CELL_SIZE: float,
        # RESIDUAL_P: float,
        # RESIDUAL_OTHERS: float
        ) -> List:

    entries = [
            ("0/U", ["Uinlet"], f"({wind_speed} 0 0)"),
            ("system/meshDict", ["surfaceFile"], "constant/triSurface/background3.stl"),
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

    return  list(y_mins) + list(objects)



def process_x_once(shape_path: str, b_box: BoundingBox) :

    shape = read_stl(shape_path)
    boxed_shape = get_boolean_boxed(shape, b_box)
    polymesh = rename_bounding_box(shape_path, boxed_shape, b_box)

    voxel = np.zeros((1, W, H, D))
    
    #볼셀크기 w h d일때
    for x in range(W):
        for y in range(H):
            for z in range(D):
    #트라이앵글은 b_box 공간에 있고 복셀은 [0,16] [0,16] [0,16] 에 정의됐으니 x z 만 평행이동해 복셀 중심을 원점으로 옮기고 /16 으로 단위 크기로 만들고 *3.0으로 트라이앵글 공간으로
                P_voxel = np.array([x-W/2, y, z-D/2])*np.array(RATIO)
                dist = 1e+9
                pt = None
    #가장 작
                for po in polymesh:
    #sdf_triangle가 복셀 좌표 P_voxel과 어떤 트라이앨글과 최단거리 반환,  루프돌면 가장 가까운거리 나옴. 
                    t = sdf_triangle(P_voxel, po.points[0], po.points[1], po.points[2])
                    if t < dist:
                        dist = t
                        pt = po
    # 가장 가까운거리를 실제 복셀데이터로. 부호를 si함수로 보정
                voxel[0][x][y][z] = dist * si(P_voxel, pt.points[0], pt.points[1], pt.points[2])

    return voxel
    

def process_y(path: str, b_box: BoundingBox, grid_size: float) -> np.array:
    sampled = spatial_sample_case_even(path, b_box, grid_size)
    u = sampled["U"]
    p = sampled["p"].reshape(-1, 1)
    k = sampled["k"].reshape(-1, 1)


    res = np.hstack([u, p, k]).reshape(256,128,256,5)
    print(res.shape)
    return res

def write_data(x, y, write_path: str):
    with open(write_path,"wb") as fw:
        pickle.dump((x, y), fw)

def get_file_name(path: str) -> str:
    return path.basename(path).split('/')[-1]


def gen_data(c: Config):  

    write_dir_path = path.join(c.local_case_dir_path, f"{i}/")
    mkdir(write_dir_path)

    path_with_speedss = get_cfd_arg_batches(c)
    
    for i, path_speeds in enumerate(path_with_speedss):

        path = path_speeds[0]
        voxel = process_x_once()

        for j, speed in enumerate(path_speeds[1]):

            entries = get_entries(c, path, speed)

            if j == 0:
                run_case(c, entries, 120)
            else:
                run_case_no_meshing(c, entries, 120)

            x = (speed, voxel)
            y = process_y(path, B_BOX, GRID_SIZE)

            write_path = path.join(write_dir_path, f"{speed}-" + get_file_name(path))

            write_data(x, y, write_path)

            



