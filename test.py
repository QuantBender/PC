
import sys

for line in sys.stdin:
    digits = [int(d.strip()) for d in line.split()]
    for digit in digits:
        print(digit)
