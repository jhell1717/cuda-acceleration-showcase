from concurrent.futures import thread
import numpy as np
import cv2
import time
from numba import cuda, float32

GAUSSIAN_KERNEL = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], dtype=np.float32)
GAUSSIAN_KERNEL /= GAUSSIAN_KERNEL.sum()  # normalize kernel

@cuda.jit
def gaussian_blur_kernel(input_img,out_img,kernel):
    x,y, = cuda.grid(2)

    height = input_img.shape[0]
    width = input_img.shape[1]
    k_size = kernel.shape[0]
    k_half = k_size //2

    if x < height and y < width:
        val = 0.0

        for i in range(-k_half, k_half + 1):
            for j in range(-k_half, k_half + 1):
                xi = min(max(x + i, 0), height - 1)
                yj = min(max(y + j, 0), width - 1)
                val += input_img[xi, yj] * kernel[i + k_half, j + k_half]

        out_img[x, y] = val

def gpu_gaussian_blur(img):
    img = img.astype(np.float32)

    d_input = cuda.to_device(img)
    d_output = cuda.device_array_like(img)
    d_kernel = cuda.to_device(GAUSSIAN_KERNEL)

    threads_per_block = (16,16)
    blocks_per_grid =(
        (img.shape[0] + threads_per_block[0] -1) // threads_per_block[0],
        (img.shape[1] + threads_per_block[1] -1) // threads_per_block[1]
    )

    start = time.time()
    gaussian_blur_kernel[blocks_per_grid, threads_per_block](d_input, d_output, d_kernel)
    cuda.synchronize()
    end = time.time()

    print(f'GPU Gaussian Blur Time: {end-start} seconds')
    
