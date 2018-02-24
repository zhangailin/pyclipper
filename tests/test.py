import PyClipper

import numpy as np

pp = np.arange(12, dtype=np.int32).reshape(6, 2)

PyClipper.test(pp)

ppnt = np.array([-1, -2], dtype=np.int32)
PyClipper.test_intpnt(ppnt)

pp1 = np.array([
    0, 0,
    10, 0,
    10, 10,
    0, 10,
    ], dtype=np.int32).reshape(4, 2)
print("Path", pp1)
print("Orie", PyClipper.Orientation(pp1))
print("Area", PyClipper.Area(pp1))
pp2 = np.array([
    0, 0,
    0, 10,
    10, 10,
    10, 0,
    0, 0
    ], dtype=np.int32).reshape(-1, 2)
print("Path", pp2)
print("Orie", PyClipper.Orientation(pp2))
print("Area", PyClipper.Area(pp2))
pp3 = np.array([
    0, 0,
    10, 0,
    5, 10,
    ], dtype=np.int32).reshape(-1, 2)
print("Area", PyClipper.Area(pp3))
pnt1 = np.array([3, 3], dtype=np.int32)
pnt2 = np.array([5, 0], dtype=np.int32)
print("PointInPolygon1", PyClipper.PointInPolygon(pnt1, pp2))
print("PointInPolygon2", PyClipper.PointInPolygon(pnt2, pp3))



###############################
#def test(pp):
#    cp = _to_clipper_path(pp)
#    puts_c_path(cp)
#
#    pp2 = _from_clipper_path(cp)
#    print(np.asarray(pp2))
#
#    cdef Path pp3
#    pp3 = np.empty((0, 0), dtype=np.int32)
#    print(pp3.size)
#
#def test_intpnt(ppnt):
#    cpnt = _to_clipper_point(ppnt)
#    print(">>>", cpnt.X, "===",  cpnt.Y, "<<<")
#    print(5, 6, 7, 8)
#
#    ppnt2 = _from_clipper_point(cpnt)
#    print(np.asarray(ppnt2))
#
#def test_2():
#    pp1 = np.array([
#        0, 0,
#        10, 0,
#        10, 10,
#        0, 10,
#        ], dtype=np.int32).reshape(4, 2)
#
#    pp2 = np.array([
#        0, 0,
#        0, 10,
#        10, 10,
#        10, 0,
#        0, 0
#        ], dtype=np.int32).reshape(-1, 2)
#
#    paths = [pp1, pp2]
#    c_paths = _to_clipper_paths(paths)

    #print(type(c_paths))
