
from mpi4py import MPI
import time
import sys

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

if RANK == 0:
    data = int(sys.argv[1])
    COMM.send(data, dest=1, tag=0)
    print(f"Process {RANK} sent value: {data}")

elif RANK < SIZE-1:
    data = COMM.recv(source=RANK-1, tag=0)
    print(f"Process {RANK} received value: {data} from process {RANK-1}")
    data += RANK
    COMM.send(data, dest=RANK+1, tag=0)
    print(f"Process {RANK} sent value: {data}")

else:
    data = COMM.recv(source=RANK-1, tag=0)
    print(f"Process {RANK} received value: {data} from process {RANK-1}")
