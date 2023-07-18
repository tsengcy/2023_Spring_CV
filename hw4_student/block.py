import numpy as np

a = np.arange(10)
print(a)
print(a[-1])
print(a[0:])
for i in range(3):
    print(a[:-i or None])

b = np.array([1, 1, 0, 0])
c = np.array([1, 0, 0, 1])

print(b)
print(c)
print(np.logical_xor(b, c))

d = list(range(0, -10, -1))
print(d)
d = list(range(10, 0, -1))
print(d)