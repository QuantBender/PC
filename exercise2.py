
import sys
import time
from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

if RANK == 0:
    for line in sys.stdin:
        value = int(line.strip())
        for i in range(1, SIZE):
            COMM.send(value, dest=i, tag=0)
        if value < 0:
            break
        print(f"Process {RANK} received value: {value}")
        time.sleep(3)
else:
    while True:
        value = COMM.recv(source=0, tag=0)
        if value < 0:
            break
        print(f"Process {RANK} received value: {value}")
        time.sleep(3)
