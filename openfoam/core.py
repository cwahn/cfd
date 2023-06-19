from copy import deepcopy
from functools import reduce
from os import chmod, listdir, path, remove
import re
from shutil import rmtree
from subprocess import Popen, TimeoutExpired, check_output, run
from time import time
from typing import Any, List, Optional, Tuple
from attr import dataclass
from more_itertools import consume, side_effect
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from psutil import cpu_count



@dataclass
class Config:
    open_foam_img: str
    local_volum_path: str
    container_mount_path: str
    rel_case_path: str
    script_file_name: str
    log_file_name: str
    
    local_case_dir_path: str
    container_case_dir_path: str
    local_script_file_path: str
    container_script_file_path: str
    local_log_file_path: str
    container_log_file_path: str

    num_proc: int


def configure(
    open_foam_img: str,
    local_volum_path: str,
    container_mount_path: str,
    rel_case_path: str,
    script_file_name: str,
    log_file_name: str) -> Config :

    c = Config(
        open_foam_img,
        local_volum_path,
        container_mount_path,
        rel_case_path,
        script_file_name,
        log_file_name,

        path.join(local_volum_path, rel_case_path),
        path.join(container_mount_path, rel_case_path),
        path.join(local_volum_path, rel_case_path, script_file_name),
        path.join(container_mount_path, rel_case_path, script_file_name),
        path.join(local_volum_path, rel_case_path, log_file_name),
        path.join(container_mount_path, rel_case_path, log_file_name),

        cpu_count(logical=True),
    )

    return c


def is_numeric_dir_name(string:str)-> bool:
    matches = re.match("^\d+$", string)
    return matches != None


def is_processor_dir_name(string: str) -> bool:
    matches = re.match("^processor\d+$", string)
    return matches != None


def is_exact_file_name(string: str, file_name: str) -> bool: 
    return string == file_name


def clear_case(c: Config):
    contents = listdir(c.local_case_dir_path)

    numeric_dirs = list(filter(is_numeric_dir_name, contents))
    time_dirs =  deepcopy(numeric_dirs)
    if "0" in time_dirs:
        time_dirs.remove("0")
    
    proc_dirs = list(filter(is_processor_dir_name, contents))
    
    log_files = list(filter(lambda s: is_exact_file_name(s, "log"), contents))

    time_dir_paths = list(map(lambda x: path.join(c.local_case_dir_path, x), time_dirs))
    proc_dir_paths = list(map(lambda x: path.join(c.local_case_dir_path, x), proc_dirs))
    log_file_paths = list(map(lambda x: path.join(c.local_case_dir_path, x), log_files))

    consume(side_effect(rmtree, time_dir_paths))
    consume(side_effect(rmtree, proc_dir_paths))
    consume(side_effect(remove, log_file_paths))

    if "constant" in contents:
        cs = listdir(path.join(c.local_case_dir_path, "constant/"))
        if "polyMesh" in cs:
            # print("Removing polyMesh")
            rmtree(path.join(c.local_case_dir_path, "constant/polyMesh"))

    # if path.isfile(c.local_script_file_path):
    #     remove(c.local_script_file_path)


def get_multi(data: Any, keys: List[Any]) -> Any:
    return reduce(lambda data, key: data[key], keys, data)


def get_dict_entry(file_path: str, entry_path: List[str]) -> Any:
    f=ParsedParameterFile(file_path)
    return get_multi(f, entry_path)


def set_multi(data: Any, keys: List[Any], value: Any) -> Any:
    init = keys[:-1]
    last = keys[-1]
    reduce(lambda data, key: data[key], init, data).__setitem__(last, value)


def set_dict_entry(file_path: str, entry_path: List[str], value: Any) -> Any:
    f=ParsedParameterFile(file_path)
    set_multi(f, entry_path, value)
    f.writeFile()


def set_case(c: Config, entries: List[Tuple[str, List[str], Any]]) -> None:
    abs_entries = list(map(lambda p: (path.join(c.local_case_dir_path, p[0]), p[1], p[2]), entries))
    consume(side_effect(lambda p: set_dict_entry(p[0], p[1], p[2]), abs_entries))

def time_process(cmd: List[str]) -> Optional[Tuple[int, float]]:
    start = time()

    with Popen(cmd) as p:
        p.communicate()

        retcode = p.poll()
    
    end = time()
    duration = end-start
    print("Retcode: ", retcode)
    print("Time: ", duration)
    return (retcode, duration)


def run_openfoam_docker(c: Config):
    run(["docker", "run", "--rm", "-itd", "-u", "1000", f"--volume={c.local_volum_path}:{c.container_mount_path}", f"{c.open_foam_img}"])


def assert_docker_id(c: Config) -> str:
    docker_ps = str(check_output(["docker", "ps"]), "utf-8")
    matches = re.findall(f"\n(.*)\s+{c.open_foam_img}*", docker_ps)

    if len(matches) == 0:
        run_openfoam_docker(c)
        return assert_docker_id(c)
    else:
        return matches[0].strip()


def get_foam_app(case_path: str) -> str:
    file_path = path.join(case_path, "system/controlDict")
    return get_dict_entry(file_path, ["application"])


def write_foam_run(c: Config):
    lines = [
        r"#!/bin/bash",
        "cartesianMesh >> ./log;",
        "decomposePar >> ./log;",
        f"mpirun -np {c.num_proc} simpleFoam -parallel >> ./log;",
        "reconstructPar >> ./log;" 
    ]

    # with open(c.local_script_file_path, "w") as f:
    #     f.writelines(map(lambda l: l+"\n", lines))

    # chmod(c.local_script_file_path, 777)

def run_all(c: Config, timeout_s=int) -> Optional[float]:
    docker_id = assert_docker_id(c)
    foam_app = get_foam_app(c.local_case_dir_path)
    _ = write_foam_run(c)

    cmd = [
        "docker",
        "exec",
        "-w",
        c.container_case_dir_path,
        docker_id,
        "timeout",
        f"{timeout_s}",
        "bash",
        "-lc",
        f'"./{c.script_file_name}"'
    ]

    result = time_process(cmd)
    
    if result != None:
        if result[0] == 0:
            return result[1]
        else:
            return None
    else:
        return None
    

def run_case(c: Config, entries: List[Tuple[str, List[str], Any]], timeout_s: int) -> Optional[Tuple[str, float]]:
    
    clear_case(c)
    
    set_case(c, entries)
    
    res = run_all(c, timeout_s)

    if res != None:
        return (c.local_case_dir_path, res)
    else:
        return None