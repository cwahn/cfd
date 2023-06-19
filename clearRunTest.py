from openfoam.core import NUM_PROC, run_case

case_path = "/Users/chanwooahn/openfoam/OpenFOAM/openfoam-2212/run/testCase/"

params = [
    ("system/controlDict", ["application"], "simpleFoam"),
    ("system/fvSolution", ["SIMPLE", "residualControl", "p"], 0.02)
    ("system/decomposeParDict", ["numberOfSubdomains"], NUM_PROC)
]

if __name__ == "__main__":
    print("Clear, Set, Run OpenFoam Case")
    res = run_case(case_path, params)
    print(res)