import numpy as np
import cv2
import time
from numba import cuda

GAUSSIAN_KERNEL = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], dtype=np.float32)
GAUSSIAN_KERNEL /= GAUSSIAN_KERNEL.sum()

# Shared-memory GPU kernel
@cuda.jit
def gaussian_blur_shared(input_img, output_img, kernel):
    # Thread coordinates in image
    x, y = cuda.grid(2)

    height = input_img.shape[0]
    width = input_img.shape[1]
    k_size = kernel.shape[0]
    k_half = k_size // 2

    # Shared memory tile with halo
    shared = cuda.shared.array(shape=(16+4, 16+4), dtype=np.float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Load image tile into shared memory
    # Handle out-of-bounds by clamping
    sx = min(max(x, 0), height - 1)
    sy = min(max(y, 0), width - 1)

    shared[tx + 2, ty + 2] = input_img[sx, sy]

    # Load halo (border pixels)
    if tx < 2 and x >= 2:
        shared[tx, ty + 2] = input_img[x - 2, sy]
    if tx >= 14 and x + 2 < height:
        shared[tx + 4, ty + 2] = input_img[x + 2, sy]
    if ty < 2 and y >= 2:
        shared[tx + 2, ty] = input_img[sx, y - 2]
    if ty >= 14 and y + 2 < width:
        shared[tx + 2, ty + 4] = input_img[sx, y + 2]

    cuda.syncthreads()

    if x < height and y < width:
        val = 0.0
        for i in range(k_size):
            for j in range(k_size):
                val += shared[tx + i, ty + j] * kernel[i, j]

        output_img[x, y] = val


def compare_gaussian_blur(img):
    img = img.astype(np.float32)

    # GPU Setup
    d_input = cuda.to_device(img)
    d_output = cuda.device_array_like(img)
    d_kernel = cuda.to_device(GAUSSIAN_KERNEL)

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (img.shape[0] + 15) // 16,
        (img.shape[1] + 15) // 16
    )

    # GPU Timing
    start_gpu = time.time()
    gaussian_blur_shared[blocks_per_grid, threads_per_block](d_input, d_output, d_kernel)
    cuda.synchronize()
    end_gpu = time.time()

    print(f"GPU Shared-Memory Blur:   {end_gpu - start_gpu:.5f} sec")

    return d_output.copy_to_host()
