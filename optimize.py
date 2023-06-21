from copy import deepcopy
from math import isnan, sqrt
from os import path
from typing import List
from openfoam.core import run_case

from openfoam.optimizer import T, last
from openfoam.post import rms_of_sampled
from openfoam.post import spatial_sample_case_even
from openfoam.graph import to_csv
from openfoam.core import configure

ref_file_path = "/home/ubuntu/openfoam/OpenFOAM/openfoam-v2212/run/refCaseSolved3/open.foam"
# ref_file_path = "/Users/chanwooahn/openfoam/OpenFOAM/openfoam-2212/run/refCaseSolved3/open.foam"

b_box = [-500, 500, 1e-2, 500, -500, 500]

grid_size = 3.9

sampled_ref = spatial_sample_case_even(ref_file_path, b_box, 3.9)

OPENFOAM_IMG = "opencfd/openfoam-default"
LOCAL_VOL_PATH = "/home/ubuntu/openfoam/"
CONTAINER_MNT_PATH = "/home/sudofoam/"
REL_CASE_PATH = "OpenFOAM/openfoam-v2212/run/optCase/"
SCRIPT_FILE_NAME = "run.sh"
LOG_FILE_NAME = "log"


# OPENFOAM_IMG = "gerlero/openfoam-default"
# LOCAL_VOL_PATH = "/Users/chanwooahn/openfoam/"
# CONTAINER_MNT_PATH = "/home/openfoam/"
# REL_CASE_PATH = "OpenFOAM/openfoam-2212/run/optCase/"
# SCRIPT_FILE_NAME = "run.sh"
# LOG_FILE_NAME = "log"


c = configure(
    OPENFOAM_IMG,
    LOCAL_VOL_PATH,
    CONTAINER_MNT_PATH,
    REL_CASE_PATH,
    SCRIPT_FILE_NAME,
    LOG_FILE_NAME,
)

# * Need rel shape file

print(c)

# maxCellSize, boundaryCellSize, localRefCellSize, residualP, residualOthers

FLOAT_MACHINE_EPSILON = 2.22e-16
FLOAT_EPSILON = sqrt(FLOAT_MACHINE_EPSILON)

class FloatAdam(T):
    def __init__(
            self, inner: float, 
            alpha: float, 
            beta1: float = 0.9, 
            beta2: float=0.99
            ) -> None:
        
        super().__init__()
        self.inner = inner
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.m1 = 0.
        self.m2 = 0.

    def succ(self) -> "FloatAdam":
        new = deepcopy(self)
        new.inner = self.inner + (self.inner * 0.05)
        return new

    def partial_diff(self, y: float, succ_y: float) -> float:
        pd = (succ_y - y)/(self.inner * 0.05)
        return pd
    
    def update(self, pds: List[float]) -> T:
        pd = last(pds)
        m1_ = self.beta1 * self.m1 + (1-self.beta1) * pd
        m2_ = self.beta2 * self.m2 + (1-self.beta2) * (pd**2)
        inner_ = self.inner - self.alpha * (m1_ / sqrt(m2_ + FLOAT_EPSILON))

        new = deepcopy(self)
        new.m1 = m1_
        new.m2 = m2_
        new.inner = inner_

        return new

    def __repr__(self):
        return f"Float<inner: {self.inner}>"


def opt_run(ts: List[T]) -> float:
    maxCellSize, boundaryCellSize, localRefCellSize, residualP, residualOthers = tuple(ts)

    entries = [
        ("0/U", ["Uinlet"], "(10 0 0)"),
        ("system/meshDict", ["surfaceFile"], "constant/triSurface/background3.stl"),
        ("system/meshDict", ["maxCellSize"], maxCellSize.inner),
        ("system/meshDict", ["boundaryCellSize"], boundaryCellSize.inner),
        ("system/meshDict", ["localRefinement", "object.*", "cellSize"], localRefCellSize.inner),
        ("system/meshDict", ["boundaryLayers", "patchBoundaryLayers", "object.*", "nLayers"], 5),
        ("system/meshDict", ["boundaryLayers", "patchBoundaryLayers", "object.*", "thicknessRatio"], 1.2),
        ("system/fvSolution", ["SIMPLE", "residualControl", "p"], residualP.inner),
        ("system/fvSolution", ["SIMPLE", "residualControl", "U"], residualOthers.inner),
        ("system/fvSolution", ["SIMPLE", "residualControl", "k"], residualOthers.inner),
        ("system/fvSolution", ["SIMPLE", "residualControl", "omega"], residualOthers.inner),
        ("system/fvSolution", ["SIMPLE", "residualControl", "epsilon"], residualOthers.inner),
        ("system/decomposeParDict", ["numberOfSubdomains"], c.num_proc),
    ]
    print("Theta: ", ts)
    res = run_case(c, entries, 60)

    if res == None:
        return 60 * 1
    
    _, time = res
    foam_file_path = path.join(c.local_case_dir_path, "open.foam")
    sampled = spatial_sample_case_even(foam_file_path, b_box, grid_size)
    error_raw = rms_of_sampled(sampled, sampled_ref)

    if error_raw < 0.1:
        error = 1
    else:
        error = error_raw * 10

    loss = time * error
    
    print(f"Error: {error}, Time: {time}, loss: {loss} \n")

    return time * error




def stop_if_small_y(_thetas, ys, _pdss) -> bool:
    return last(ys) < 0.01

def keep_go(_thetas, _ys, _pdss) -> bool:
    return False

# def log_by_print(i, ts, y, pds):
#     print("Interation: ", i)
#     print("Theta: ", ts)
#     print("y: ", y)
#     print("pds: ",pds, "\n")

logger = to_csv(
    "./optimize.csv", 
    [
        "maxCellSize", 
        "boundaryCellSize", 
        "localRefCellSize", 
        "residualP", 
        "residualOthers", 
        "y", 
        "maxCellSizeD", 
        "boundaryCellSizeD", 
        "localRefCellSizeD", 
        "residualPD", 
        "residualOthersD"
    ])

from openfoam.optimizer import optimize


theta0s = [
    [FloatAdam(100., 1, 0.8, 0.88)],
    [FloatAdam(50., 1, 0.8, 0.88)],
    [FloatAdam(5., 1, 0.8, 0.88)],
    [FloatAdam(5e-1,0.01, 0.8, 0.88)],
    [FloatAdam(5e-2, 0.01, 0.8, 0.88)]
]

pdss = [[], [], [], [], []]

optimize(
    opt_run,
    theta0s,
    [],
    pdss,
    keep_go,
    logger
)