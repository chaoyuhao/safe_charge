from sca3_game import sca3_step

ba = [1,1,1,1,1]

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
