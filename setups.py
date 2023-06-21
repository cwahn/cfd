import torch
import numpy as np


from pyvista import OpenFOAMReader

from openfoam.post import *
from openfoam.prep import *
import trimesh


# FILE_PATH = "./data/results/windAroundBuildingsOld/open.foam"
FILE_PATH = "./testCase/open.foam"
MESH_PATH = "./testCase/constant/triSurface/box-in-box.stl"

def dot2(x):
    return np.dot(x, x)

# numpy.array
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



def save_and_rename_bounding_box(
        path: str,
        boxed_object: pv.PolyData,
        b_box: BoundingBox,
):
    #temp_path = path.replace(".stl", "_.stl")

    #save_stl(boxed_object, temp_path)

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



class Dataset:
    def __init__(self,w,h,d,dataset_size):
        """
        :w,h,d: width, height, depth (sizes in x,y,z direction)
        :types: possibilities: "no_box","box","rod_y","rod_z","moving_rod_y","moving_rod_z","magnus_y","magnus_z","ball","image","benchmark"
        :images: if type is image, then there are the following possibilities: see img_dict
        """



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
