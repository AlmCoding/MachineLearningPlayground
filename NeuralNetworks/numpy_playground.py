import numpy as np


a = np.array([[1, 2, 3]]).transpose()
print('a:', a, a.shape)

a_rep = np.repeat(a, 2)
print('a_repeat:', a_rep)
a_tile = np.tile(a, (2, 2))
print('a_tile:', a_tile)

b = np.array([[1, 2, 3]])
print(b, b.shape)