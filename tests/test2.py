import numpy as np
import clipperx as clx

p1 = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0], dtype=np.int32)
path_1 = clx.Path(p1)

print("p1:", p1)
print("path_1:", path_1)
print("path_1[2]", path_1[2])
print("path_1[1:3]", path_1[1:3])
#print("path_1[7]", path_1[7])
print("path_1['a']", path_1['a'])
for p in path_1:
    print(p)
