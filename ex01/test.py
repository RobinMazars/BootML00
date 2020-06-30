from matrix import Matrix
from pprint import pprint
from vector import Vector
import numpy as np
m1 = Matrix((2, 3))
m2 = Matrix([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0]])
m3 = Matrix([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0],
             [4.0, 5.0, 6.0, 7.0]], (3, 4))
v1 = Vector([1.0, 1.0, 2.0, 3.0])
arr = [[0.0, 1.0, 2.0, 3.0],
       [0.0, 2.0, 4.0, 6.0]]

narr = [[0.0, 1.0],
        [1.0, 2.0],
        [4.0, 5.0],
        [10.0, 20.0]]

mtest = Matrix(arr)
tm = np.array(arr)
tm2 = np.array(narr)

mtest2 = Matrix(narr)
vtest = Vector([1., 2., 3., 4.])
# pprint(mtest)
# pprint(mtest2)
# pprint(m2 + m2)
# # pprint(v1 + m2)
# pprint(m2 + v1)
# pprint(m2.__radd__(v1))
# pprint(m2 - m2)
# pprint(m2 - v1)
# pprint(m2.__rsub__(v1))
# pprint(m2 / 2)
# pprint(m2.__rtruediv__(2))
# pprint(mtest * vtest)
# pprint(mtest * mtest)
# print(n.dot(2))
# print(m1.__mul__(2))
# print()
# print(n.dot(n1))
# print(m1.__mul__(v))
# print()
# print(n.dot(np.array(narr)))
pprint(mtest.__mul__(mtest2))
pprint(mtest.__rmul__(mtest2))
print(tm.dot(tm2))
print(tm.dot(tm2))
