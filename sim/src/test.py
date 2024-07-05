from sca3_game import sca3_step
from sca3_game import query_max_k
import numpy
ba = [1,1,1,1,1]

print('max open chargers:', query_max_k())

a = sca3_step(ba)

print(a)

ba[0] = 0

b = sca3_step(ba)

print(b)

ba[0] = 1
ba[1] = 0

c = sca3_step(ba)

print(c)

ba[1] = 1
ba[2] = 0

d = sca3_step(ba)

print(d)

chargers_num = 5
map_status = [0,0] * (2**chargers_num-1)
args_lamda = 500

print(d[0] - args_lamda * d[1])
def ba2ctrl(x):
    base = 0
    for i in range(x):
        base = (base<<1) + x[i]
    return base

reward  = []
penalty = []
def matrix_gen(x):
    global ba
    if x >= chargers_num:
        idx = ba2ctrl(ba)
        reward[idx], penalty[idx] = sca3_step(ba)
        return

    ba[x] = 1
    matrix_gen(x+1)
    ba[x] = 0
    matrix_gen(x+1)
    

