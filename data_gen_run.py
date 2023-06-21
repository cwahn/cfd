
from openfoam.data_gen import gen_data
from openfoam.core import configure


OPENFOAM_IMG = "opencfd/openfoam-default"
LOCAL_VOL_PATH = "/home/ubuntu/openfoam/"
CONTAINER_MNT_PATH = "/home/sudofoam/"
REL_CASE_PATH = "OpenFOAM/openfoam-v2212/run/dataGen/"
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

FIRST_RUN = True

gen_data(c, FIRST_RUN)