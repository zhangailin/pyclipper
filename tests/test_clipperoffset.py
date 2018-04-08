import numpy as np
import clipperx as clx
import copy

path1 = clx.Path(np.array([0, 0,
                           0, 10,
                           10, 10], dtype=np.int64))

co = clx.ClipperOffset()
co.add_path(path1,
            clx.JT_SQUARE,
            clx.ET_OPENBUTT)
plist = co.execute(2)
print(plist)
