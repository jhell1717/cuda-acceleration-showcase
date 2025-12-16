from numba import cuda, vectorize
import numpy as np
import time


@cuda.jit
def matrix_multiply_kernel(A, B, C):
    """CUDA kernel for matrix multiplication C = A @ B"""
    x, y = cuda.grid(2)
    
    if x < C.shape[0] and y < C.shape[1]:
        val = 0.0
        for i in range(A.shape[1]):
            val += A[x, i] * B[i, y]
        C[x, y] = val

def gpu_matrix_multiply(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    #Specify dimensions to match original image.
    d_C = cuda.device_array((A.shape[0], B.shape[1]), dtype=np.float32)

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (A.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
        (B.shape[1] + threads_per_block[1] - 1) // threads_per_block[1])

    start = time.time()
    matrix_multiply_kernel[blocks_per_grid, threads_per_block](d_A,d_B,d_C)
    cuda.synchronize()
    end = time.time()

    print(f'GPU Matrix Multiplication Time: {end-start} seconds')



@vectorize(['float32(float32,float32)'],device='cuda')
def matrix_gpu(A,B):
    return A*B