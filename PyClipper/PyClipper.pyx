from libcpp cimport bool
from libcpp.vector cimport vector
from libc cimport stdlib

cimport numpy as cnp
import numpy as np


cnp.import_array()


SCALING_FACTOR = 1

cdef extern from "clipper.hpp" namespace "ClipperLib":
     # define use_int32 1 in setup.py
     ctypedef int cInt
     ctypedef signed long long long64
     ctypedef unsigned long long ulong64

     # enum ClipType { ctIntersection, ctUnion, ctDifference, ctXor };
     cdef enum ClipType:
         ctIntersection = 1
         ctUnion = 2
         ctDifference = 3
         ctXor = 4

     # enum PolyType { ptSubject, ptClip };
     cdef enum PolyType:
         ptSubject = 1
         ptClip = 2

     # enum PolyFillType { pftEvenOdd, pftNonZero, pftPositive, pftNegative };
     cdef enum PolyFillType:
         pftEvenOdd = 1
         pftNonZero = 2
         pftPositive = 3
         pftNegative = 4

     #enum InitOptions {ioReverseSolution = 1, ioStrictlySimple = 2, ioPreserveCollinear = 4};
     cdef enum InitOptions:
         ioReverseSolution = 1
         ioStrictlySimple = 2
         ioPreserveCollinear = 4

     #enum JoinType { jtSquare, jtRound, jtMiter };
     cdef enum JoinType:
         jtSquare = 1
         jtRound = 2
         jtMiter = 3

     #enum EndType {etClosedPolygon, etClosedLine, etOpenButt, etOpenSquare, etOpenRound};
     cdef enum EndType:
         etClosedPolygon = 1
         etClosedLine = 2
         etOpenButt = 3
         etOpenSquare = 4
         etOpenRound = 5

     cdef struct c_IntPoint 'ClipperLib::IntPoint':
         cInt X
         cInt Y

     ctypedef vector[c_IntPoint] c_Path 'ClipperLib::Path'
     ctypedef vector[c_Path] c_Paths 'ClipperLib::Paths'
     ctypedef vector[c_PolyNode*] c_PolyNodes 'ClipperLib::PolyNodes'

     cdef cppclass c_PolyNode 'ClipperLib::PolyNode':
         c_PolyNode()
         c_Path Contour
         c_PolyNodes Childs
         c_PolyNode *Parent
         c_PolyNode *GetNext()
         bool IsHole()
         bool IsOpen()
         int ChildCount()

     cdef cppclass c_PolyTree 'ClipperLib::PolyTree':
         c_PolyTree()
         c_PolyNode &GetFirst()
         void Clear()
         int Total()

     cdef struct c_IntRect 'ClipperLib::IntRect':
         cInt left
         cInt top
         cInt right
         cInt bottom

     cdef cppclass c_Clipper 'ClipperLib::Clipper':
         c_Clipper(int initOptions)
         c_Clipper()
         void Clear()
         bool Execute(ClipType clipType, c_Paths  &solution, PolyFillType subjFillType, PolyFillType clipFillType)
         bool Execute(ClipType clipType, c_PolyTree  &solution, PolyFillType subjFillType, PolyFillType clipFillType)
         bool ReverseSolution()
         void ReverseSolution(bool value)
         bool StrictlySimple()
         void StrictlySimple(bool value)
         bool PreserveCollinear()
         void PreserveCollinear(bool value)
         bool AddPath(c_Path  &path, PolyType polyType, bool closed)
         bool AddPaths(c_Paths  &paths, PolyType polyType, bool closed)
         c_IntRect GetBounds()

     cdef cppclass c_ClipperOffset 'ClipperLib::ClipperOffset':
         c_ClipperOffset(double miterLimit, double roundPrecision)
         c_ClipperOffset(double miterLimit)
         c_ClipperOffset()
         void AddPath(c_Path  &path, JoinType joinType, EndType endType)
         void AddPaths(c_Paths  &paths, JoinType joinType, EndType endType)
         void Execute(c_Paths  &solution, double delta)
         void Execute(c_PolyTree  &solution, double delta)
         void Clear()
         double MiterLimit
         double ArcTolerance

     bool c_Orientation "ClipperLib::Orientation"(const c_Path &poly)
     double c_Area "ClipperLib::Area"(const c_Path &poly)
     int c_PointInPolygon "ClipperLib::PointInPolygon"(const c_IntPoint &pt, const c_Path &path)

     void c_SimplifyPolygon "ClipperLib::SimplifyPolygon"(const c_Path &in_poly, c_Paths &out_polys, PolyFillType fillType)
     void c_SimplifyPolygons "ClipperLib::SimplifyPolygons"(const c_Paths &in_polys, c_Paths &out_polys, PolyFillType fillType)
     void c_CleanPolygon "ClipperLib::CleanPolygon"(const c_Path &in_poly, c_Path &out_poly, double distance)
     void c_CleanPolygons "ClipperLib::CleanPolygons"(c_Paths &polys, double distance)

     void c_MinkowskiSum "ClipperLib::MinkowskiSum"(const c_Path &pattern, const c_Path &path, c_Paths &solution, bool pathIsClosed)
     void c_MinkowskiSum "ClipperLib::MinkowskiSum"(const c_Path &pattern, const c_Paths &paths, c_Paths &solution, bool pathIsClosed)
     void c_MinkowskiDiff "ClipperLib::MinkowskiDiff"(const c_Path &poly1, const c_Path &poly2, c_Paths &solution)

     void c_PolyTreeToPaths "ClipperLib::PolyTreeToPaths"(const c_PolyTree &polytree, c_Paths &paths)
     void c_ClosedPathsFromPolyTree "ClipperLib::ClosedPathsFromPolyTree"(const c_PolyTree &polytree, c_Paths &paths)
     void c_OpenPathsFromPolyTree "ClipperLib::OpenPathsFromPolyTree"(c_PolyTree &polytree, c_Paths &paths)

     void c_ReversePath "ClipperLib::ReversePath"(c_Path &p)
     void c_ReversePaths "ClipperLib::ReversePaths"(c_Paths &p)
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


ctypedef int[:] IntPoint
ctypedef int[:, ::1] Path


def puts_c_path(c_Path path):
    for i in range(path.size()):
        print(path.at(i).X, path.at(i).Y)


cdef IntPoint _from_clipper_point(c_IntPoint c_point):
    return <int[:2]>(<int*>&c_point)


cdef c_IntPoint _to_clipper_point(IntPoint point):
    if point.ndim != 1 or point.size != 2:
        raise ValueError("point must be 1d and size == 0")
    return (<c_IntPoint*>&point[0])[0]


cdef Path _from_clipper_path(c_Path c_path):
    cdef int *raw_ptr = <int*>&(c_path[0])
    dim0 = c_path.size()
    return <int[:dim0, :2]>raw_ptr


cdef c_Path _to_clipper_path(Path path):
    cdef c_IntPoint *raw_ptr = <c_IntPoint*>&path[0][0]
    cdef vector[c_IntPoint] vec
    cdef int len = path.size // 2
    vec.assign(raw_ptr, raw_ptr + len)
    return vec


cdef c_Paths _to_clipper_paths(list paths):
    cdef c_Paths c_paths = c_Paths()
    for path in paths:
        c_paths.push_back(_to_clipper_path(path))
    return c_paths


cdef list _from_clipper_paths(c_Paths c_paths):
    cdef list polys = []

    cdef c_Path c_path
    for i in range(c_paths.size()):
        c_path = c_paths[i]
        polys.append(_from_clipper_path(c_path))

    return polys


cdef PolyNode _from_clipper_polytree(c_PolyTree &c_polytree):
    cdef PolyNode polytree = PolyNode()
    cdef int i
    cdef c_PolyNode* c_child
    cdef PolyNode py_child
    cdef int depth_a1
    cdef int depth = 0

    cdef c_PolyNode *pt_base = <c_PolyNode*>&c_polytree

    for i in range(pt_base.ChildCount()):
        c_child = pt_base.Childs[i]
        py_child = _node_walk(c_child, polytree)
        polytree.Childs.append(py_child)

        depth_a1 = py_child.depth + 1
        depth = depth_a1 if depth_a1 > depth else depth

    polytree.depth = depth

    return polytree


cdef PolyNode _node_walk(c_PolyNode* c_polynode, PolyNode parent):
    cdef PolyNode polynode = PolyNode()
    polynode.Parent = parent
    polynode.IsHole = <bint>c_polynode.IsHole()
    polynode.IsOpen = <bint>c_polynode.IsOpen()
    polynode.Contour = _from_clipper_path(c_polynode.Contour)

    # kids
    cdef c_PolyNode *c_node
    cdef PolyNode py_child
    cdef int i
    cdef int depth_a1
    cdef int depth = 0
    for i in range(c_polynode.ChildCount()):
        c_node = c_polynode.Childs[i]
        py_child = _node_walk(c_node, polynode)
        polynode.Childs.append(py_child)

        depth_a1 = py_child.depth + 1
        depth = depth_a1 if depth_a1 > depth else depth

    polynode.depth = depth

    return polynode


cdef _filter_polynode(PolyNode polynode, list result, filter_func=None):
    if (filter_func is None or filter_func(polynode)) and polynode.Contour.size > 0:
        result.append(polynode.Contour)

    for child in polynode.Childs:
        _filter_polynode(child, result, filter_func)


#------------------------------ ClipperException -----------------------------
class ClipperException(Exception):
    pass


#------------------------------ IntRect -----------------------------
cdef class IntRect:
    cdef:
        public int left
        public int top
        public int right
        public int bottom
    def __cinit__(IntRect self,
            int left,
            int top,
            int right,
            int bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


#------------------------------ PolyNode -----------------------------
cdef class PolyNode:
    cdef:
        public Path Contour
        public list Childs
        public PolyNode Parent
        public bint IsHole
        public bint IsOpen
        public int depth

    def __cinit__(self):
        self.Contour = np.empty((0, 2), dtype=np.int32)
        self.Childs = []
        self.Parent = None
        self.IsHole = False
        self.IsOpen = False
        self.depth = 0


cpdef bint Orientation(Path poly):
    """ Get orientation of the supplied polygon.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/Orientation.htm

    Keyword arguments:
    poly -- closed polygon
    Returns:
    True  -- counter-clockwise orientation
    False -- clockwise orientation
    """
    return <bint>c_Orientation(_to_clipper_path(poly))


cpdef double Area(Path poly):
    """ Get area of the supplied polygon.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/Area.htm

    Keyword arguments:
    poly -- closed polygon

    Returns:
    Positive number if orientation is True
    Negative number if orientation is False
    """

    return <double>c_Area(_to_clipper_path(poly))


cpdef int PointInPolygon(IntPoint point, Path poly):
    """ Determine where does the point lie regarding the provided polygon.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/PointInPolygon.htm

    Keyword arguments:
    point -- point in question
    poly  -- closed polygon

    Returns:
    0  -- point is not in polygon
    -1 -- point is on polygon
    1  -- point is in polygon
    """

    return <int>c_PointInPolygon(_to_clipper_point(point),
            _to_clipper_path(poly))


cpdef list SimplifyPolygon(Path poly, PolyFillType fill_type=pftEvenOdd):
    """ Removes self-intersections from the supplied polygon.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/SimplifyPolygon.htm

    Keyword arguments:
    poly      -- polygon to be simplified
    fill_type -- PolyFillType used with the boolean union operation

    Returns:
    list of simplified polygons (containing one or more polygons)
    """
    cdef c_Paths out_polys
    c_SimplifyPolygon(_to_clipper_path(poly), out_polys, fill_type)
    return _from_clipper_paths(out_polys)


cpdef list SimplifyPolygons(list polys, PolyFillType fill_type=pftEvenOdd):
    """ Removes self-intersections from the supplied polygons.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/SimplifyPolygons.htm

    Keyword arguments:
    polys     -- polygons to be simplified
    fill_type -- PolyFillType used with the boolean union operation

    Returns:
    list of simplified polygons
    """
    cdef c_Paths out_polys
    c_SimplifyPolygons(_to_clipper_paths(polys), out_polys, fill_type)
    return _from_clipper_paths(out_polys)


cpdef Path CleanPolygon(Path poly, double distance=1.415):
    """ Removes unnecessary vertices from the provided polygon.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/CleanPolygon.htm

    Keyword arguments:
    poly     -- polygon to be cleaned
    distance -- distance on which vertices are removed, see 'More info' (default: approx. sqrt of 2)

    Returns:
    cleaned polygon
    """
    cdef c_Path out_poly
    c_CleanPolygon(_to_clipper_path(poly), out_poly, distance)
    return _from_clipper_path(out_poly)


cpdef list CleanPolygons(list polys, double distance=1.415):
    """ Removes unnecessary vertices from the provided polygons.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/CleanPolygons.htm

    Keyword arguments:
    polys    -- polygons to be cleaned
    distance -- distance on which vertices are removed, see 'More info' (default: approx. sqrt of 2)

    Returns:
    list of cleaned polygons
    """
    cdef c_Paths out_polys = _to_clipper_paths(polys)
    c_CleanPolygons(out_polys, distance)
    return _from_clipper_paths(out_polys)


cpdef list MinkowskiSum(Path pattern, Path path, bint path_is_closed):
    """ Performs Minkowski Addition of the pattern and path.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/MinkowskiSum.htm

    Keyword arguments:
    pattern        -- polygon whose points are added to the path
    path           -- open or closed path
    path_is_closed -- set to True if passed path is closed, False if open

    Returns:
    list of polygons (containing one or more polygons)
    """
    cdef c_Paths solution
    c_MinkowskiSum(_to_clipper_path(pattern),
                 _to_clipper_path(path),
                 solution,
                 path_is_closed)
    return _from_clipper_paths(solution)


cpdef list MinkowskiSum2(Path pattern, list paths, bint path_is_closed):
    """ Performs Minkowski Addition of the pattern and paths.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/MinkowskiSum.htm

    Keyword arguments:
    pattern        -- polygon whose points are added to the paths
    paths          -- open or closed paths
    path_is_closed -- set to True if passed paths are closed, False if open

    Returns:
    list of polygons
    """
    cdef c_Paths solution
    c_MinkowskiSum(
        _to_clipper_path(pattern),
        _to_clipper_paths(paths),
        solution,
        path_is_closed
    )
    return _from_clipper_paths(solution)


cpdef list MinkowskiDiff(Path poly1, Path poly2):
    """ Performs Minkowski Difference.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/MinkowskiDiff.htm

    Keyword arguments:
    poly1 -- polygon
    poly2 -- polygon

    Returns:
    list of polygons
    """
    cdef c_Paths solution
    c_MinkowskiDiff(_to_clipper_path(poly1), _to_clipper_path(poly2), solution)
    return _from_clipper_paths(solution)


cpdef list PolyTreeToPaths(PolyNode poly_node):
    """ Converts a PolyNode to a list of paths.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/PolyTreeToPaths.htm

    Keyword arguments:
    poly_node -- PolyNode to be filtered

    Returns:
    list of paths
    """
    cdef list paths = []
    _filter_polynode(poly_node, paths, filter_func=None)
    return paths


### cpdef does not support closure
cdef list _ClosedPathsFromPolyTree(PolyNode poly_node):
    """ Filters out open paths from the PolyNode and returns only closed paths.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/ClosedPathsFromPolyTree.htm

    Keyword arguments:
    poly_node -- PolyNode to be filtered

    Returns:
    list of closed paths
    """

    cdef list paths = []
    _filter_polynode(poly_node, paths, filter_func=lambda pn: not pn.IsOpen)
    return paths


def ClosedPathsFromPolyTree(PolyNode poly_node):
    return _ClosedPathsFromPolyTree(poly_node)


cdef list _OpenPathsFromPolyTree(PolyNode poly_node):
    """ Filters out closed paths from the PolyNode and returns only open paths.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/OpenPathsFromPolyTree.htm

    Keyword arguments:
    poly_node -- PolyNode to be filtered

    Returns:
    list of open paths
    """
    cdef list paths = []
    _filter_polynode(poly_node, paths, filter_func=lambda pn: pn.IsOpen)
    return paths


def OpenPathsFromPolyTree(PolyNode poly_node):
    return _OpenPathsFromPolyTree(poly_node)


cpdef Path ReversePath(Path path):
    """ Reverses the vertex order (and hence orientation) in the specified path.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/ReversePath.htm

    Keyword arguments:
    path -- path to be reversed

    Returns:
    reversed path
    """
    cdef c_Path c_path = _to_clipper_path(path)
    c_ReversePath(c_path)
    return _from_clipper_path(c_path)


cpdef list ReversePaths(list paths):
    """ Reverses the vertex order (and hence orientation) in each path.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/ReversePaths.htm

    Keyword arguments:
    paths -- paths to be reversed

    Returns:
    list if reversed paths
    """
    cdef list result = []
    for p in paths:
        result.append(ReversePath(p))

    return result


cdef class Clipper:
    """Wraps the cpp Clipper class.

    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/_Body.htm
    """
    cdef c_Clipper *thisptr  # hold a C++ instance which we're wrapping
    def __cinit__(Clipper self):
        """ Creates an instance of the Clipper class. InitOptions from the Clipper class
        are substituted with separate properties.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Methods/Constructor.htm
        """

        self.thisptr = new c_Clipper()

    def __dealloc__(Clipper self):
        del self.thisptr

    cpdef AddPath(Clipper self, Path path, PolyType poly_type, bint closed=True):
        """ Add individual path.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperBase/Methods/AddPath.htm

        Keyword arguments:
        path      -- path to be added
        poly_type -- type of the added path - subject or clip
        closed    -- True if the added path is closed, False if open

        Returns:
        True -- path is valid for clipping and was added

        Raises:
        ClipperException -- if path is invalid for clipping
        """
        cdef c_Path c_path = _to_clipper_path(path)
        cdef bint result = <bint>self.thisptr.AddPath(c_path, poly_type, closed)
        if not result:
            raise ClipperException('The path is invalid for clipping')
        return result

    cpdef AddPaths(Clipper self, list paths, PolyType poly_type, bint closed=True):
        """ Add a list of paths.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperBase/Methods/AddPaths.htm

        Keyword arguments:
        paths     -- paths to be added
        poly_type -- type of added paths - subject or clip
        closed    -- True if added paths are closed, False if open

        Returns:
        True -- all or some paths are valid for clipping and were added

        Raises:
        ClipperException -- all paths are invalid for clipping
        """
        cdef c_Paths c_paths = _to_clipper_paths(paths)
        cdef bint result = <bint>self.thisptr.AddPaths(c_paths, poly_type, closed)
        if not result:
            raise ClipperException('All paths are invalid for clipping')
        return result

    cpdef Clear(Clipper self):
        """ Removes all subject and clip polygons.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperBase/Methods/Clear.htm
        """
        self.thisptr.Clear()

    cpdef GetBounds(Clipper self):
        """ Returns an axis-aligned bounding rectangle that bounds all added polygons.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperBase/Methods/GetBounds.htm

        Returns:
        IntRect with left, right, bottom, top vertices that define the axis-aligned bounding rectangle.
        """

        cdef c_IntRect rr = <c_IntRect>self.thisptr.GetBounds()

        return IntRect(left=rr.left, top=rr.top,
                right=rr.right, bottom=rr.bottom)

    cpdef Execute(Clipper self,
            ClipType clip_type,
            PolyFillType subj_fill_type=pftEvenOdd,
            PolyFillType clip_fill_type=pftEvenOdd):
        """ Performs the clipping operation and returns a list of paths.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Methods/Execute.htm

        Keyword arguments:
        clip_type      -- type of the clipping operation
        subj_fill_type -- fill rule of subject paths
        clip_fill_type -- fill rule of clip paths

        Returns:
        list of resulting paths

        Raises:
        ClipperException -- operation did not succeed
        """

        cdef c_Paths solution
        cdef bint success = <bint>self.thisptr.Execute(clip_type, solution, subj_fill_type, clip_fill_type)
        if not success:
            raise ClipperException('Execution of clipper did not succeed!')
        return _from_clipper_paths(solution)

    cpdef Execute2(Clipper self,
            ClipType clip_type,
            PolyFillType subj_fill_type=pftEvenOdd,
            PolyFillType clip_fill_type=pftEvenOdd):
        """ Performs the clipping operation and returns a PyPolyNode.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Methods/Execute.htm

        Keyword arguments:
        clip_type      -- type of the clipping operation
        subj_fill_type -- fill rule of subject paths
        clip_fill_type -- fill rule of clip paths

        Returns:
        PyPolyNode

        Raises:
        ClipperException -- operation did not succeed
        """
        cdef c_PolyTree solution
        cdef bint success = <bint>self.thisptr.Execute(clip_type, solution, subj_fill_type, clip_fill_type)
        if not success:
            raise ClipperException('Execution of clipper did not succeed!')
        return _from_clipper_polytree(solution)

    property ReverseSolution:
        """ Should polygons returned from Execute/Execute2 have their orientations
        opposite to their normal orientations.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Properties/ReverseSolution.htm
        """
        def __get__(self):
            return <bint> self.thisptr.ReverseSolution()

        def __set__(self, bint value):
            self.thisptr.ReverseSolution(value)

    property PreserveCollinear:
        """ Should clipper preserve collinear vertices.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Properties/PreserveCollinear.htm
        """
        def __get__(self):
            return <bint> self.thisptr.PreserveCollinear()

        def __set__(self, bint value):
            self.thisptr.PreserveCollinear(value)

    property StrictlySimple:
        """ Should polygons returned from Execute/Execute2 be strictly simple (True) or may be weakly simple (False).
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Properties/StrictlySimple.htm
        """
        def __get__(self):
            return <bint> self.thisptr.StrictlySimple()

        def __set__(self, bint value):
            self.thisptr.StrictlySimple(value)


cdef class ClipperOffset:
    """ Wraps the cpp ClipperOffset class.

    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/_Body.htm
    """
    cdef c_ClipperOffset *thisptr

    def __cinit__(self, double miter_limit=2.0, double arc_tolerance=0.25):
        """ Creates an instance of the ClipperOffset class.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/Constructor.htm
        """
        self.thisptr = new c_ClipperOffset(miter_limit, arc_tolerance)

    def __dealloc__(self):
        del self.thisptr

    cpdef AddPath(self, Path path, JoinType join_type, EndType end_type):
        """ Add individual path.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/AddPath.htm

        Keyword arguments:
        path      -- path to be added
        join_type -- join type of added path
        end_type  -- end type of added path
        """
        cdef c_Path c_path = _to_clipper_path(path)
        self.thisptr.AddPath(c_path, join_type, end_type)

    cpdef AddPaths(self, list paths, JoinType join_type, EndType end_type):
        """ Add a list of paths.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/AddPaths.htm

        Keyword arguments:
        path      -- paths to be added
        join_type -- join type of added paths
        end_type  -- end type of added paths
        """
        cdef c_Paths c_paths = _to_clipper_paths(paths)
        self.thisptr.AddPaths(c_paths, join_type, end_type)

    cpdef Execute(self, double delta):
        """ Performs the offset operation and returns a list of offset paths.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/Execute.htm

        Keyword arguments:
        delta -- amount to which the supplied paths will be offset - negative delta shrinks polygons,
                 positive delta expands them.

        Returns:
        list of offset paths
        """
        cdef c_Paths c_solution
        self.thisptr.Execute(c_solution, delta)
        return _from_clipper_paths(c_solution)

    cpdef Execute2(self, double delta):
        """ Performs the offset operation and returns a PyPolyNode with offset paths.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/Execute.htm

        Keyword arguments:
        delta -- amount to which the supplied paths will be offset - negative delta shrinks polygons,
                 positive delta expands them.

        Returns:
        PyPolyNode
        """
        cdef c_PolyTree c_solution
        self.thisptr.Execute(c_solution, delta)
        return _from_clipper_polytree(c_solution)

    cpdef Clear(self):
        """ Clears all paths.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/Clear.htm
        """
        self.thisptr.Clear()

    property MiterLimit:
        """ Maximum distance in multiples of delta that vertices can be offset from their
        original positions.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Properties/MiterLimit.htm
        """
        def __get__(self):
            return <double>self.thisptr.MiterLimit

        def __set__(self, double value):
            self.thisptr.MiterLimit = value

    property ArcTolerance:
        """ Maximum acceptable imprecision when arcs are approximated in
        an offsetting operation.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Properties/ArcTolerance.htm
        """
        def __get__(self):
            return self.thisptr.ArcTolerance

        def __set__(self, double value):
            self.thisptr.ArcTolerance = value
