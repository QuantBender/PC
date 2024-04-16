
# Import the necessary libraries
import numpy as np
import random
from mpi4py import MPI

seed = 10
random.seed(seed)
np.random.seed(seed)

# Initialize the MPI communication
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# matrix size
matrix_size = 4

# local matrix size
local_matrix_size = matrix_size // SIZE if RANK != SIZE - 1 else matrix_size // SIZE + matrix_size % SIZE


#initialize matrix A in rank 0
if RANK == 0:
    A = np.array([np.random.randint(0,1000) for _ in range(matrix_size*matrix_size)]).reshape(matrix_size, matrix_size).astype(np.int64)
    A_diagonal = np.diagonal(A, offset=0).astype(np.int64)
else:
    A_diagonal = None

local_A_diagonal = np.empty((local_matrix_size, 1), dtype=np.int64)
    
# Scatter the diagonal
COMM.Scatter(A_diagonal, [local_A_diagonal, local_matrix_size], root=0)

# Compute the sum of the diagonal
local_sum = np.sum(local_A_diagonal)

# Reduce the sum of the diagonal
global_sum = COMM.reduce(local_sum, op=MPI.SUM, root=0)

# Print the result in rank 0
if RANK == 0:
    print(f"Matrix A:\n{A}")
    print(f"Sum of the diagonal: {global_sum}")
    

