from cuda import cuda, cudart, nvrtc
import numpy as np


from openfoam.prep import *


X = 256
Y = 128
Z = 256
SIZE = X*Y*Z



AAAAA =  """\

#include "helper_math.h"

#define cl_device __device__

cl_device float sign( float x )
{
  return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
}


cl_device float dot2( float3 x )
{
  return dot(x, x);
}

cl_device float sdf_triangle( float3 p, float3 *a, float3 *b, float3 *c )
{
  float3 ba = *b - *a; float3 pa = p - *a;
  float3 cb = *c - *b; float3 pb = p - *b;
  float3 ac = *a - *c; float3 pc = p - *c;
  float3 nor = cross( ba, ac );

  return sqrt(
    (sign(dot(cross(ba,nor),pa)) +
     sign(dot(cross(cb,nor),pb)) +
     sign(dot(cross(ac,nor),pc))<2.0)
     ?
     min( min(
     dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
     dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
     dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc) )
     :
     dot(nor,pa)*dot(nor,pa)/dot2(nor) );
}
    
extern "C" 
{ 

__global__ void voxelize(
    float *vert,
    int facenum,
    float *mesh)
{

    uint3 id = blockDim*blockIdx + threadIdx;
    uint3 dim = make_uint3(256, 128, 256);//blockDim*gridDim;
    
    float3 voxel = (make_float3(id.x, id.y, id.z) - make_float3(128,0,128))      * make_float3(1000,500,1000) / make_float3(256,128,256);
    int idx = 0;
    float dist = 1e+9;

    for (int i = 0; i < facenum; ++i) {
        int n = i*9;

        float t = sdf_triangle(voxel, 
            (float3*)(&vert[n]), 
            (float3*)(&vert[n+3]),
            (float3*)(&vert[n+6])
        );

        if (t < dist) {
            dist = t;
            idx = n;
        }
    }

    (mesh)[id.x + id.y*dim.x + id.z*dim.x *dim.y] = dist ;  

}
} // extern C
"""






def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))
    
def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
    




def calc_sdf(kernel, stream, stl_path):

    #상대경로?
    # stl_path = './refCaseSolved3/constant/triSurface/background3.stl'
    polymesh = parse_stl_file(stl_path)


    pointss = list(map(lambda x: x.points, polymesh))
    points = np.array(pointss).ravel()

    # 인자들의 cpu 쪽 
    hbuv = np.array(points+[0.0]).astype(dtype=np.float32)

    hbulen = np.array([(len(points)-1)/9]).astype(dtype=np.int32)

    hOut = np.zeros(SIZE, dtype=np.float32)


    # 바이트 크기
    floatsize = np.int32(4)



    #  gpu 메모리 할당

    err, dVclass = cuda.cuMemAlloc(len(hbuv)*floatsize)
    ASSERT_DRV(err)

    err, dOutclass = cuda.cuMemAlloc(SIZE * floatsize)
    ASSERT_DRV(err)





    # 할당한 메모리로 복사
    ASSERT_DRV(err)
    err, = cuda.cuMemcpyHtoDAsync(
    dVclass, hbuv.ctypes.data, len(hbuv)*floatsize, stream
    )
    ASSERT_DRV(err)
    err, = cuda.cuMemcpyHtoDAsync(
    dOutclass, hOut.ctypes.data, SIZE * floatsize, stream
    )
    ASSERT_DRV(err)



    # 인자로 넣을 준비

    # The following code example is not intuitive 
    # Subject to change in a future release
    dY = np.array([int(dVclass)], dtype=np.uint64)


    dOut = np.array([int(dOutclass)], dtype=np.uint64)

    args = [dY, hbulen, dOut]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

    err, = cuda.cuLaunchKernel(
        kernel,
        X/8,  # grid x dim
        Y/8,  # grid y dim
        Z/8,  # grid z dim
        8,  # block x dim
        8,  # block y dim
        8,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream
        args.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )
    ASSERT_DRV(err)

    err, = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)

    # 결과를 cpu에 있는 hOut 으로
    err, = cuda.cuMemcpyDtoHAsync(
        hOut.ctypes.data, dOutclass, SIZE *4, stream
    )
    ASSERT_DRV(err)

    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)


    # 인자로 슨 메모리 해제   오브젝트마다 크기다르기에 재사용 ㄴ
    err, = cuda.cuMemFree(dVclass)
    err, = cuda.cuMemFree(dOutclass)

    return hOut









# paths 는 리스트 오브 stl의 경로    처리할 stl의 경로를 리스트로 담음
def get_sdfs(paths):
    # import os

    # CUDA_HOME = os.getenv('CUDA_HOME')
    # if CUDA_HOME == None:
    #     CUDA_HOME = os.getenv('CUDA_PATH')
    # if CUDA_HOME == None:
    #     raise RuntimeError('Environment variable CUDA_HOME or CUDA_PATH is not set')
    # include_dirs = os.path.join(CUDA_HOME, 'include')


    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(AAAAA), b"sdfpy.cu", 0, [], [])

    # Compile program 
    # cuda_runtime.h 라는 파일에 경로   /usr/local/cuda-1xxxxxxx/  이하 경로 어딘가  
    PATH = '/usr/local/cuda-12.1/targets/x86_64-linux/include'

    opts = ["--include-path={} ".format(PATH).encode('UTF-8')]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

    # compile err log
    logSize = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b' ' * logSize
    checkCudaErrors(nvrtc.nvrtcGetProgramLog(prog, log))
    print(log.decode())
    ASSERT_DRV(err)

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ASSERT_DRV(err)
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    ASSERT_DRV(err)

    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)
    # Retrieve handle for device 0
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)
    # Create context
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ASSERT_DRV(err)


    # Load PTX as module data and retrieve function
    ptx = np.char.array(ptx)
    # Note: Incompatible --gpu-architecture would be detected here
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(module, b"voxelize")
    ASSERT_DRV(err)

    err, stream = cuda.cuStreamCreate(0)
    ASSERT_DRV(err)

    mesh_sd = np.array([])

    for path in paths:
        mesh_sd =   np.append(mesh_sd,       calc_sdf(kernel, stream, path)         )



    err, = cuda.cuStreamDestroy(stream)

    err, = cuda.cuModuleUnload(module)
    err, = cuda.cuCtxDestroy(context)


    return mesh_sd



