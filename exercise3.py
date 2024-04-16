
from mpi4py import MPI
import time

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

n = 0

while n!=8:
    if RANK % 2 == 0:
        COMM.send(n, dest=RANK+1, tag=0)
        # print(f"Process {RANK} sent value: {n}")
        n = COMM.recv(source=RANK+1, tag=0)
        print(f"Process {RANK} received value: {n} from process {RANK+1}")
        time.sleep(3)
    else:
        n = COMM.recv(source=RANK-1, tag=0)
        print(f"Process {RANK} received value: {n} from process {RANK-1}")
        n += 1
        COMM.send(n, dest=RANK-1, tag=0)
        # print(f"Process {RANK} sent value: {n}")
