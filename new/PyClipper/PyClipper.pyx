from libcpp cimport bool
from libcpp.vector cimport vector
cimport stdlib

cimport numpy as cnp


cnp.install_array()

cdef extern from "clipper.hpp" namespace "ClipperLib":
     # define use_int32 1 in setup.py
     ctypedef int cInt
     ctypedef signed long long long64
     ctypedef unsigned long long ulong64

     # enum ClipType { ctIntersection, ctUnion, ctDifference, ctXor };
     cdef enum ClipType:
         ctIntersection = 1,
         ctUnion = 2,
         ctDifference = 3,
         ctXor = 4

     # enum PolyType { ptSubject, ptClip };
     cdef enum PolyType:
         ptSubject = 1,
         ptClip = 2

     # enum PolyFillType { pftEvenOdd, pftNonZero, pftPositive, pftNegative };
     cdef enum PolyFillType:
         pftEvenOdd = 1,
         pftNonZero = 2,
         pftPositive = 3,
         pftNegative = 4

     cdef struct c_IntPoint 'ClipperLib::IntPoint':
         cInt X
         cInt Y
         c_IntPoint(cInt x, cInt y)

     #typedef std::vector< IntPoint > Path;
     cdef cppclass c_Path 'ClipperLib::Path':
         c_Path()
         void push_back(IntPoint &)
         IntPoint &operator[](int)
         IntPoint &at(int)
         int size()

     #typedef std::vector< Path > Paths;
     cdef cppclass c_Paths 'ClipperLib::Paths':
         c_Paths()
         void push_back(c_Path &)
         c_Path &operator[](int)
         c_Path &at(int)
         int size()

     cdef cppclass c_PolyNode 'ClipperLib::PolyNode':
         c_PolyNode()
         c_Path Contour
         c_PolyNodes Childs
         c_PolyNode *Parent
         c_PolyNode *GetNext()
         bool IsHole()
         bool IsOpen()
         int ChildCount()

     cdef cppclass c_PolyNodes 'ClipperLib::PolyNodes':
         c_PolyNodes()
         void push_back(c_PolyNode &)
         c_PolyNode *operator[](int)
         c_PolyNode *at(int)
         int size()

     cdef cppclass c_PolyTree 'ClipperLib::PolyTree':
         c_PolyTree()
         c_PolyNode& GetFirst()
         void Clear()
         int Total()

     #enum InitOptions {ioReverseSolution = 1, ioStrictlySimple = 2, ioPreserveCollinear = 4};
     cdef enum InitOptions:
         ioReverseSolution = 1,
         ioStrictlySimple = 2,
         ioPreserveCollinear = 4

     #enum JoinType { jtSquare, jtRound, jtMiter };
     cdef enum JoinType:
         jtSquare = 1,
         jtRound = 2,
         jtMiter = 3

     #enum EndType {etClosedPolygon, etClosedLine, etOpenButt, etOpenSquare, etOpenRound};
     cdef enum EndType:
         etClosedPolygon = 1,
         etClosedLine = 2,
         etOpenButt = 3,
         etOpenSquare = 4,
         etOpenRound = 5

     cdef struct c_IntRect 'ClipperLib::IntRect':
         cInt left
         cInt top
         cInt right
         cInt bottom

     cdef cppclass c_Clipper 'ClipperLib::Clipper':
         c_Clipper(int initOptions)
         c_Clipper()
         #~c_Clipper()
         void Clear()
         bool Execute(ClipType clipType, c_Paths & solution, PolyFillType subjFillType, PolyFillType clipFillType)
         bool Execute(ClipType clipType, c_PolyTree & solution, PolyFillType subjFillType, PolyFillType clipFillType)
         bool ReverseSolution()
         void ReverseSolution(bool value)
         bool StrictlySimple()
         void StrictlySimple(bool value)
         bool PreserveCollinear()
         void PreserveCollinear(bool value)
         bool AddPath(c_Path & path, PolyType polyType, bool closed)
         bool AddPaths(c_Paths & paths, PolyType polyType, bool closed)
         c_IntRect GetBounds()

     cdef cppclass c_ClipperOffset 'ClipperLib::ClipperOffset':
         c_ClipperOffset(double miterLimit, double roundPrecision)
         c_ClipperOffset(double miterLimit)
         c_ClipperOffset()
         #~c_ClipperOffset()
         void AddPath(c_Path & path, JoinType joinType, EndType endType)
         void AddPaths(c_Paths & paths, JoinType joinType, EndType endType)
         void Execute(c_Paths & solution, double delta)
         void Execute(c_PolyTree & solution, double delta)
         void Clear()
         double MiterLimit
         double ArcTolerance

     bool c_Orientation "ClipperLib::Orientation"(const c_Path &poly)
     double c_Area "ClipperLib::Area"(const c_Path &poly)
     int c_PointInPolygon "ClipperLib::PointInPolygon"(const c_IntPoint & pt, const c_Path &path)

     void c_SimplifyPolygon "ClipperLib::SimplifyPolygon"(const c_Path & in_poly, c_Paths & out_polys, PolyFillType fillType)
     void c_SimplifyPolygons "ClipperLib::SimplifyPolygons"(const c_Paths & in_polys, c_Paths & out_polys, PolyFillType fillType)
     void c_CleanPolygon "ClipperLib::CleanPolygon"(const c_Path &in_poly, c_Path &out_poly, double distance)
     void c_CleanPolygons "ClipperLib::CleanPolygons"(c_Paths& polys, double distance)

     void c_MinkowskiSum "ClipperLib::MinkowskiSum"(const c_Path& pattern, const c_Path& path, c_Paths& solution, bool pathIsClosed)
     void c_MinkowskiSum "ClipperLib::MinkowskiSum"(const c_Path& pattern, const c_Paths& paths, c_Paths& solution, bool pathIsClosed)
     void c_MinkowskiDiff "ClipperLib::MinkowskiDiff"(const c_Path& poly1, const c_Path& poly2, c_Paths& solution)

     void c_PolyTreeToPaths "ClipperLib::PolyTreeToPaths"(const c_PolyTree& polytree, c_Paths& paths)
     void c_ClosedPathsFromPolyTree "ClipperLib::ClosedPathsFromPolyTree"(const c_PolyTree& polytree, c_Paths& paths)
     void c_OpenPathsFromPolyTree "ClipperLib::OpenPathsFromPolyTree"(c_PolyTree& polytree, c_Paths& paths)

     void c_ReversePath "ClipperLib::ReversePath"(c_Path& p)
     void c_ReversePaths "ClipperLib::ReversePaths"(c_Paths& p)
### end of 'cdef extern from ...'

#------------------------------ Enum mapping -----------------------------
JT_SQUARE = jtSquare
JT_ROUND = jtRound
JT_MITER = jtMiter

ET_CLOSEDPOLYGON = etClosedPolygon
ET_CLOSEDLINE = etClosedLine
ET_OPENBUTT = etOpenButt
ET_OPENSQUARE = etOpenSquare
ET_OPENROUND = etOpenRound

CT_INTERSECTION = ctIntersection
CT_UNION = ctUnion
CT_DIFFERENCE = ctDifference
CT_XOR = ctXor

PT_SUBJECT = ptSubject
PT_CLIP = ptClip

PFT_EVENODD = pftEvenOdd
PFT_NONZERO = pftNonZero
PFT_POSITIVE = pftPositive
PFT_NEGATIVE = pftNegative


cdef class _finalizer:
    cdef void *_data

    def __cinit__(self):
        self._data = NULL

    def __dealloc__(self):
        if self._data != NULL:
            stdlib.free(self._data)

cdef void set_base(cnp.ndarray arr, void *carr):
    cdef _finalizer f = _finalizer()
    f._data = <void*>carr
    cnp.set_array_base(arr, f)

cdef cnp.ndarray np_from_c_data(void *data, int nd, cnp.npy_intp *dims, dtype_enum):
    if data == NULL:
        raise MemoryError()
    if nd <= 0:
        raise ValueError("nd must be >= 0, got %d".format(nd))
    arr = cnp.PyArray_SimpleNewFromData(nd, dims, dtype_enum, data)
    dobj = _finalizer()
    dobj._data = data
    cnp.set_array_base(arr, dobj)
    return arr

##XXX:
cdef np.ndarray _from_clipper_paths(Paths paths):

    polys = []

    cdef Path path
    for i in xrange(paths.size()):
        path = paths[i]
        polys.append(_from_clipper_path(path))

    return polys

##XXX:
cdef object _from_clipper_path(Path path):
    _check_scaling_factor()

    poly = []
    cdef IntPoint point
    for i in xrange(path.size()):
        point = path[i]
        poly.append([point.X, point.Y])
    return poly

##XXX:
cdef c_IntPoint _to_clipper_point(object py_point):
    return c_IntPoint(py_point[0], py_point[1])

#------------------------------ IntPoint -----------------------------
##XXX:
cdef class IntPoint:
    cdef:
         cInt X
         cInt Y
         c_IntPoint *c_ptr

    def __cinit__(self, cInt x=0, cInt y=0):
        self.c_ptr = new c_IntPoint(x, y)
        self.X = x
        self.Y = y

    def __dealloc__(self):
        del self.c_ptr


#------------------------------ Path -----------------------------
##XXX:
cdef class Path:
    cdef:
        c_Path *c_ptr
        c_Path()
        void push_back(IntPoint &)
        IntPoint &operator[](int)
        IntPoint &at(int)
        int size()


#------------------------------ ClipperException -----------------------------
class ClipperException(Exception):
    pass
#------------------------------ IntRect -----------------------------
cdef class IntRect:
    cdef:
        int left
        int top
        int right
        int bottom



#------------------------------ PolyNode -----------------------------
##XXX:
cdef class PolyNode:
    cdef:
        Path Contour
        PolyNodes Childs
        PolyNode Parent
        bool IsHole
        bool IsOpen
        int depth

    def __init__(self):
        self.Contour = []
        self.Childs = []
        self.Parent = None
        self.IsHole = False
        self.IsOpen = False
        self.depth = 0



