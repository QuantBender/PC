
# Import requirements
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize shared comm utils
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Initial condition function
def f(x):
    if (300 <= x <= 400):
        return 10
    return 0

# Bar Length & speed
L = 1000
c = 1

# Space mesh
nx = 200
dx = L / (nx-1)
x = np.linspace(0, L, nx)

# Time mesh
CFL = 1
nt = 110
dt = CFL * (dx/abs(c))

# Init
u = np.array([f(x[i]) for i in range(nx)])
un = np.zeros(nx)
    
# Solver
def solve_1d_linearconv(u, un, nt, nx, dt, dx, c):
    for n in range(nt):  
        # Iterative update
        un = u.copy()
        for i in range(1, nx): 
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
            
        # Processes interchange and periodic boundary conditions (u[0] = u[nx-1])
        # COMM.send(u[-1], dest=(RANK+1)%SIZE, tag=2)
        # u[0] = COMM.recv(source=(RANK-1)%SIZE, tag=2)
        u[0] = COMM.sendrecv(u[-1], dest=(RANK+1)%SIZE, sendtag=2, source=(RANK-1)%SIZE, recvtag=2)
        
    return 0

# ------------------------------------------------
# Parallel paradigm: Let solve and plot solution
# ------------------------------------------------

assert SIZE <= nx and SIZE >= 2, "The number of process should not be greater than total mesh size"

if RANK == 0:
    exec_time = time.time()

    # Let Split u array
    u_array = np.array_split(u, SIZE)
    new_u_array = [np.concatenate(([u_array[i-1][-1]], u_array[i])) for i in range(1, len(u_array))]
    new_u_array.insert(0, u_array[0])
    
    # Send u to each process
    for process in range(SIZE):
        COMM.send(new_u_array[process], dest=process, tag=0)

# Wait for data and send result to P0
u = COMM.recv(source=0, tag=0)
nx = len(u)
un = np.zeros(nx)
solve_1d_linearconv(u, un, nt, nx, dt, dx, c)
COMM.send(u, dest=0, tag=1)

if RANK == 0:
    # Collect and join all chunch of u
    u_final = np.array([])
    for process in range(SIZE):
        u_chunck = COMM.recv(source=process, tag=1)
        u_final = np.concatenate((u_final, u_chunck)) if process == 0 else np.concatenate((u_final, u_chunck[1:]))
    
    # Time
    print(F"Execution time: {time.time() - exec_time}s")

    # Plot u
    plt.figure()
    plt.grid()
    plt.plot(x, u_final, '-')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title("1D Linear Convection Visualization")
    plt.show()
