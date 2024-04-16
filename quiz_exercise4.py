
import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

import time, random

from mpi4py import MPI


# initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

seed = 10
np.random.seed(seed)
random.seed(seed)


def solve_2d_diff(u, un, nt, dt, dx, dy, nu):
    row, col =u.shape

    ###Assign initial conditions
    # set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    u[int(.5 / dy):int(1 / dy +1),int(.5 / dx):int(1 / dx +1)] =2
    #fill the update of u
    for n in range(nt + 1):
        un = u.copy()
        row, col = u.shape
        for i in range(1, row):
            for j in range(1, col):
                u[i, j] = (un[i, j] + nu * dt / dx**2 * (un[i + 1, j] - 2 * un[i, j]\
                    + un[i - 1, j]) + nu * dt / dy**2 * (un[i, j + 1] - 2 * un[i, j]\
                        + un[i, j - 1]))             

        # c) effectuer les communications non bloquantes nécessaires pour échanger les valeurs manquantes
        # create the requests for the non-blocking communication
        reqs = []
        # send upwards
        req = comm.Isend(local_u[-2, :], dest=north)
        reqs.append(req)
        # send downwards
        req = comm.Isend(local_u[1, :], dest=south)
        reqs.append(req)
        # send to the left
        req = comm.Isend(local_u[:, -2], dest=east)
        reqs.append(req)
        # send to the right
        req = comm.Isend(local_u[:, 1], dest=west)
        reqs.append(req)
        # receive from upwards
        req = comm.Irecv(local_u[0, :], source=north)
        reqs.append(req)
        # receive from downwards
        req = comm.Irecv(local_u[-1, :], source=south)
        reqs.append(req)
        # receive from the left
        req = comm.Irecv(local_u[:, 0], source=east)
        reqs.append(req)
        # receive from the right
        req = comm.Irecv(local_u[:, -1], source=west)
        reqs.append(req)
        # wait for all the non-blocking communication to finish
        MPI.Request.waitall(reqs)
    return 0

###variable declarations
nt = 51
nx = 101
ny = 101
nu = .05
dx = 2 / (nx -1)
dy = 2 / (ny -1)
sigma = .25
dt = sigma * dx * dy / nu

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx)) # create a 1xn vector of 1's
un = np.ones((ny, nx))


# a) diviser le domaine en p parties, chacune gérée par un processus différent  avec Create_cart
# b) chaque processus calcul la condition initiale dans son sous domaine
# c) effectuer les communications non bloquantes nécessaires pour échanger les valeurs manquantes 
# d) le processus de rang 0 récupère toutes les résultats locaux pour afficher la solution globale.
# e) vérifier que le résultat est le même quelque soit le nombre de processus.

# a) diviser le domaine en p parties, chacune gérée par un processus différent avec Create_cart
cart2d = comm.Create_cart([size//2, size//2], periods=(False, False), reorder=False)

# b) chaque processus calcul la condition initiale dans son sous domaine
# determine the coordinates of the process in the cartesian grid
coords = cart2d.Get_coords(rank)
# determine the neighbors
north, south = cart2d.Shift(0, 1)
east, west = cart2d.Shift(1, 1)
# determine the size of the subdomain + offset
local_nx = nx // size + 2
local_ny = ny // size + 2
# determine the starting and ending indices of the subdomain
startx = coords[0] * local_nx + 1
endx = startx + local_nx - 1
starty = coords[1] * local_ny + 1
endy = starty + local_ny - 1
# create the local arrays
local_u = np.ones((local_ny, local_nx))
local_un = np.ones((local_ny, local_nx))
# set the initial conditions
local_u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2.0

# solve_2d_diff(u, un, nt, dt, dx, dy, nu) locally
solve_2d_diff(local_u, local_un, nt, dt, dx, dy, nu)

# d) le processus de rang 0 récupère toutes les résultats locaux pour afficher la solution globale.
# create an array to store the solution
u = None
if rank == 0:
    u = np.empty((ny, nx))
# gather all the local arrays to the global array u
comm.Gather(local_u, u, root=0)



fig = pyplot.figure(figsize=(7, 5), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u, cmap=cm.viridis)
