import pickle
import numpy as np
from sympy import Tuple
import torch


from openfoam.post import spatial_sample_case_even

from openfoam.prep import read_stl, save_and_rename_bounding_box
from setups import sdf_triangle


def read_data(data_path: str) :
    with open(data_path, "br") as f:
        return pickle.load(f)
    

class gitDataset:
    def __init__(self, grid_size: Tuple[int, int, int], dataset_size):
        # """
        # :w,h,d: width, height, depth (sizes in x,y,z direction)
        # :types: possibilities: "no_box","box","rod_y","rod_z","moving_rod_y","moving_rod_z","magnus_y","magnus_z","ball","image","benchmark"
        # :images: if type is image, then there are the following possibilities: see img_dict
        # """

        u, v, w = grid_size

        b_box = [-1.5, 1.5, 0, 3, -1.5, 1.5]

        
        tmp= read_stl(MESH_PATH)
        polymesh = save_and_rename_bounding_box(MESH_PATH, tmp, b_box)

        self.mesh = torch.zeros(dataset_size,1,w,h,d)

        self.ref = torch.zeros(dataset_size,5*w*h*d)


        amp = spatial_sample_case_even(FILE_PATH, b_box, 0.1875)
        U = amp["U"]
        u = np.zeros(shape=(3,16,16,16))
        u[0] = U[:,0].reshape((16, 16, 16))
        u[1] = U[:,1].reshape((16, 16, 16))
        u[2] = U[:,2].reshape((16, 16, 16))
        u = u[:].ravel()

        #u /= u.mean()

        p = amp["p"]
        #p /= p.mean()
        k = amp["k"]
        #k /= k.mean()
        
        
        fs = [u, p, k]
        flats = list(map(lambda x: np.ravel(x), fs))
        flat =  torch.Tensor(np.concatenate(flats))



        for data in range(dataset_size):
            print(data)
            self.ref[data] = flat

            for x in range(w):
                for y in range(h):
                    for z in range(d):
                        P_voxel = np.array([x-8, y, z-8])*3.0/16.0
                        dist = 1e+9
                        pt = None
                        for po in polymesh:
                            t = sdf_triangle(P_voxel, po.points[0], po.points[1], po.points[2])
                            if t < dist:
                                dist = t
                                pt = po

                        self.mesh[data][0][x][y][z] = dist * si(P_voxel, pt.points[0], pt.points[1], pt.points[2])


        self.w,self.h,self.d = w,h,d

        self.dataset_size = dataset_size
        


        
    def fetch(self, index):
        
        return self.mesh, self.ref
