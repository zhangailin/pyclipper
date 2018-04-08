import numpy as np
import clipperx as clx
import copy

p1 = np.array([[180, 200], [260, 200], [260, 150], [180, 150]], dtype=np.int32)
path_1 = clx.Path(p1)

print("p1:", p1)
print("path_1:", path_1)
print("path_1[2]", path_1[2])
print("path_1[1:3]", path_1[1:3])
# print("path_1[7]", path_1[7])
# print("path_1['a']", path_1['a'])
for p in path_1:
    print(p)
print("path_1.orientation:", path_1.orientation())
path_1_rev = clx.Path(p1[::-1])
print("path_1_rev:", path_1_rev)
print("path_1_rev.orientation:", path_1_rev.orientation())
path_1_rev2 = path_1[::-1]
print("path_1_rev2:", path_1_rev2)
print("path_1_rev2.orientation:", path_1_rev2.orientation())
path_1_rev3 = path_1_rev2[::-1]
print("path_1_rev3:", path_1_rev3)
print("path_1_rev3.orientation:", path_1_rev3.orientation())

point_1 = clx.Point(210, 170)
print("point_1:", point_1)
print("(clx.point_in_polygon(point_1, path_1)):",
      clx.point_in_polygon(point_1, path_1))
print(clx.point_in_polygon(clx.Point(300, 300), path_1))
print(clx.point_in_polygon(clx.Point(260, 180), path_1))
