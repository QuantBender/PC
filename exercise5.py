
# Libraries Import
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys, time

# MPI Initialization
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# Function for the initial conditions
def initial_conditions(x):
    return 5 if x>=2 else 0

# Parameters and variables
L = 15
nx = 500
dx = L / (nx - 1)
nt = 6
CFL = 1
c = 1
dt =(CFL * dx) / c

# Initial conditions
if RANK == 0:
    x = np.linspace(0,L,nx)
    u0 = np.array([initial_conditions(i) for i in x])
    
    local_nx = nx // SIZE
    local_x = x[RANK*local_nx:(RANK+1)*local_nx]
    local_u0 = u0[RANK*local_nx:(RANK+1)*local_nx]
    local_u = np.zeros(local_nx)
    for i in range(local_nx):
        local_u[i] = local_u0[i]
    
    # Split the u0 in arrays of size local_nx in list comprehension
    u0_split = [u0[i*local_nx:(i+1)*local_nx] if i!= SIZE-1 else u0[i*local_nx:] for i in range(SIZE)]
    local_x_split = [x[i*local_nx:(i+1)*local_nx] if i!= SIZE-1 else x[i*local_nx:] for i in range(SIZE)]
    
    # print size of each array in u0_split
    # for index, array in enumerate(u0_split):
    #     print(f"Process {index} has {len(array)} elements")
        
    # Send splitted arrays to each process
    for i in range(1, SIZE):
        COMM.send(u0_split[i], dest=i, tag=0)
        COMM.send(local_x_split[i], dest=i, tag=1)
else:
    # receive the splitted array
    local_x = COMM.recv(source=0, tag=1)
    local_u0 = COMM.recv(source=0, tag=0)
    local_nx = len(local_u0)
    local_u = np.zeros(local_nx)
    for i in range(local_nx):
        local_u[i] = local_u0[i]

if RANK == 0:
    fig, axs = plt.subplots(1, SIZE+1, figsize=(50, 20))
    

# Plot the initial solution
if RANK == 0:
    axs[0].plot(x, u0, "-r")
    axs[0].set_xlabel('Distance')
    axs[0].set_ylabel('Velocity')
    axs[0].set_title('initial', fontsize=16)
    axs[0].set_xlim([0,14])
    axs[0].set_ylim([0,6])
    axs[0].grid()

    # Plot the local initial conditions
    axs[1].plot(local_x, local_u, "-*b")
    axs[1].set_xlabel('Distance')
    axs[1].set_ylabel('Velocity')
    axs[1].set_title(f'{RANK}', fontsize=16)
    axs[1].set_xlim([0,14])
    axs[1].set_ylim([0,6])
    axs[1].grid()
else:
    # subplot
    fig, ax = plt.subplots(1, 1)
    # Plot the local initial conditions
    # print(f"Plotting process {RANK}")
    ax.plot(local_x, local_u, "-+g")
    ax.set_xlabel('Distance')
    ax.set_ylabel('Velocity')
    ax.set_title(f'{RANK}', fontsize=16)
    ax.set_xlim([0,14])
    ax.set_ylim([0,6])
    ax.grid()
    
    # send ax to process 0
    COMM.send(ax, dest=0, tag=5)
    
if RANK == 0:
    for i in range(1, SIZE):
        ax = COMM.recv(source=i, tag=5)
        lines = ax.get_lines()
        for line in lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            
        axs[i+1].plot(x_data, y_data, "-*b")
        axs[i+1].set_title(f'{i}', fontsize=16)
        axs[i+1].set_xlim([0,14])
        axs[i+1].set_ylim([0,6])
        axs[i+1].grid()
    fig.suptitle('Initial Conditions', fontsize=16)
    # update layout
    fig.tight_layout()
    fig.savefig(f'initial_conditions_split.png')
    
    
# pad local_u for processes 1 to size - 1 at the beginning with a ghost cell to hold the value at the end of the previous process local_u
if RANK != 0:
    local_u = np.pad(local_u, (1, 0), 'edge')
    
# Solve the problem
start_time = time.time()
t = 0
    
un = np.zeros(local_nx)
    
while t <= nt:  
    for i in range(local_nx):
        un[i] = local_u[i]
    for i in range(1, local_nx): 
        local_u[i] = un[i] - ((c * dt) / dx) * (un[i] - un[i-1])
    t += dt
    
    # send the value at the end of the local_u to the next process
    if RANK == 0:
        COMM.send(local_u[-1], dest=RANK+1, tag=100)
    elif RANK != SIZE-1:
        COMM.send(local_u[-1], dest=RANK+1, tag=100)
        local_u[0] = COMM.recv(source=RANK-1, tag=100)
    else:
        local_u[0] = COMM.recv(source=RANK-1, tag=100)
    # if RANK == 0:
    #     COMM.send(local_u[-1], dest=RANK+1, tag=100)
    # elif RANK != SIZE-1:
    #     local_u[0] = COMM.sendrecv(sendobj=local_u[-1], dest=RANK+1, sendtag=100, source=RANK-1, recvtag=100)
    # else:
    #     local_u[0] = COMM.recv(source=RANK-1, tag=100)
    

# Concatenate results to process 0
if RANK != 0:
    COMM.send(local_u[1:], dest=0, tag=98)
else:
    u_final = np.array([])
    u_final = np.concatenate((local_u, u_final))
    for i in range(1, SIZE):
        u_final = np.concatenate((u_final, COMM.recv(source=i, tag=98)))

if RANK == 0:
    print(f"Execution time: {time.time() - start_time}")
    # print(x)
    # print(u0)
    # print(u_final)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # Plot the final solution
    axs[0].plot(x, u0, "-r")
    axs[0].set_xlabel('Distance')
    axs[0].set_ylabel('Velocity')
    axs[0].set_title('initial', fontsize=16)
    axs[0].set_xlim([0,14])
    axs[0].set_ylim([0,6])
    axs[0].grid()
    
    axs[1].plot(x, u_final, "-b")
    axs[1].set_xlabel('Distance')
    axs[1].set_ylabel('Velocity')
    axs[1].set_title('final', fontsize=16)
    axs[1].set_xlim([0,14])
    axs[1].set_ylim([0,6])
    axs[1].grid()
    fig.suptitle('1D Linear Convection', fontsize=16)
    # update layout
    fig.tight_layout()
    fig.savefig(f'1d_linear_convection_split_{SIZE}.png')
