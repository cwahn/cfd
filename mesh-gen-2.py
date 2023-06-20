from itertools import islice, product
from more_itertools import flatten, consume,side_effect
from os import listdir, path
from random import Random, uniform

from openfoam.mesh import prepare_mesh, read_mesh

BACKGROUND_PATH = "./data/marine-city-3.obj"
DESIGNS_DIR = "./data/designs/"
OUTPUT_DIR = "./data/output/meshes/"

design_file_names = listdir(DESIGNS_DIR)
design_paths = list(map(lambda f: path.join(DESIGNS_DIR, f), design_file_names))

max_rotations = list(map(lambda f: float(f.split("-")[1].replace(".obj", "")), design_file_names))

def random(max: float) -> float:
    return uniform(0, max)

max_rotation_repeats = map(lambda f: [f]*3, max_rotations)

rotationss = map(lambda rs: list(map(random, rs)), max_rotation_repeats)

wind_dir_maxss =  map(lambda a: [a] * 36, [360] * 10)

wind_dirss = map(lambda ms: list(map(lambda m: random(m), ms)), wind_dir_maxss)

ziped = zip(design_paths, rotationss, wind_dirss)

combs = map(lambda t: list(product([t[0]], t[1], t[2])), ziped)

suffled_combs = list(flatten(combs))
Random(42).shuffle(suffled_combs)

def get_output_path(in_path:str, rel_rot: float, wind_rot: float) -> str:
    return in_path.replace(".obj", f"-{rel_rot}-{wind_rot}.stl").replace(DESIGNS_DIR, OUTPUT_DIR)

arg_with_outpaths = map(lambda t: (t[0], get_output_path(t[0], t[1], t[2]), t[1], t[2]), suffled_combs)

background =  read_mesh(BACKGROUND_PATH)

b_box = [-500, 500, 1e-2, 500, -500, 500]

# print(len(list(arg_with_outpaths)))

consume(side_effect(lambda t: prepare_mesh(t[0], t[1], background, t[2], t[3], b_box), arg_with_outpaths))