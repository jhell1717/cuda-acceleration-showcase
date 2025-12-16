import numpy as np
import time

def cpu_matrix_multiply(A,B):
    start = time.time()
    result = np.dot(A,B)
    end = time.time()

    print(f'GPU Matrix Multiplication Time: {end-start} seconds')
    return result
