"""
Cython wrapper for the C++ translation of the Angus Johnson's Clipper
library (ver. 6.2.1) (http://www.angusj.com/delphi/clipper.php)

This wrapper was written by Maxime Chalton, Lukas Treyer, Gregor Ratajc and al.

"""
import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t

cimport ClipperLib as cl

import numbers


cdef inline int64_t i64min(int64_t a, int64_t b):
    return a if a < b else b


cdef inline int64_t i64max(int64_t a, int64_t b):
    return a if a > b else b

#============================= Enum mapping ================

JT_SQUARE = cl.jtSquare
JT_ROUND  = cl.jtRound
JT_MITER  = cl.jtMiter

ET_CLOSEDPOLYGON = cl.etClosedPolygon
ET_CLOSEDLINE    = cl.etClosedLine
ET_OPENBUTT      = cl.etOpenButt
ET_OPENSQUARE    = cl.etOpenSquare
ET_OPENROUND     = cl.etOpenRound

CT_INTERSECTION = cl.ctIntersection
CT_UNION        = cl.ctUnion
CT_DIFFERENCE   = cl.ctDifference
CT_XOR          = cl.ctXor

PT_SUBJECT = cl.ptSubject
PT_CLIP    = cl.ptClip

PFT_EVENODD  = cl.pftEvenOdd
PFT_NONZERO  = cl.pftNonZero
PFT_POSITIVE = cl.pftPositive
PFT_NEGATIVE = cl.pftNegative


class ClipperException(Exception):
    pass


cdef class Point:
    cdef:
        public int64_t X
        public int64_t Y

    def __init__(self, int64_t X, int64_t Y):
        self.X, self.Y = X, Y

    def __repr__(self):
        return "Point({}, {})".format(self.X, self.Y)

    cdef cl.IntPoint to_clipper_point(self):
        return cl.IntPoint(self.X, self.Y)

    @staticmethod
    cdef Point from_clipper_point(cl.IntPoint cl_point):
        return Point(cl_point.X, cl_point.Y)

    @staticmethod
    def from_ndarray(np.ndarray array)-> Point:
        return Point(array[0], array[1])


cdef class Rect:
    cdef:
        public int64_t left
        public int64_t top
        public int64_t right
        public int64_t bottom

    def __init__(self, *,
                 int64_t left,
                 int64_t top,
                 int64_t right,
                 int64_t bottom):
        """
        Init left/top/right/bottom
        """
        self.left   = i64min(left, right)
        self.right  = i64max(left, right)
        self.top    = i64min(bottom, top)
        self.bottom = i64max(bottom, top)

    @staticmethod
    cdef Rect from_clipper_rect(cl.IntRect cl_rect):
        return Rect(left=cl_rect.left,
                    top=cl_rect.top,
                    right=cl_rect.right,
                    bottom=cl_rect.bottom)



cdef class PathList(list):
    cdef inline cl.Paths to_clipper_paths(self):
        cdef cl.Paths cl_paths = cl.Paths()
        cdef Path poly
        for poly in self:
            cl_paths.push_back(poly.cl_path())
        return cl_paths

    @staticmethod
    cdef PathList from_clipper_paths(cl.Paths cl_paths, dtype=np.int64):
        cdef PathList polys = PathList()

        cdef cl.Path cl_path
        cdef unsigned int i
        for i in range(cl_paths.size()):
            cl_path = cl_paths.at(i)
            polys.append(Path.from_clipper_path(cl_path, dtype))

        return polys

    cpdef PathList simplify(self, cl.PolyFillType fill_type=cl.pftEvenOdd):
        """ Removes self-intersections from the supplied polygons.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/SimplifyPolygons.htm

        Keyword arguments:
        fill_type -- PolyFillType used with the boolean union operation

        Returns:
        list of simplified polygons
        """
        cdef cl.Paths out_polys
        cl.SimplifyPolygons(self.to_clipper_paths(), out_polys, fill_type)
        return PathList.from_clipper_paths(out_polys)

    cpdef PathList clean(self, double distance=1.415):
        """ Removes unnecessary vertices from the provided polygons.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/CleanPolygons.htm

        Keyword arguments:
        polys    -- polygons to be cleaned
        distance -- distance on which vertices are removed, see 'More info' (default: approx. sqrt of 2)

        Returns:
        list of cleaned polygons
        """
        cdef cl.Paths out_polys = self.to_clipper_paths()
        cl.CleanPolygons(out_polys, distance)
        return PathList.from_clipper_paths(out_polys)

    cpdef PathList minkowski_sum(self, Path pattern, bint path_is_closed=True):
        """ Performs Minkowski Addition of the pattern and paths.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/MinkowskiSum.htm

        Keyword arguments:
        pattern        -- polygon whose points are added to the paths
        paths          -- open or closed paths
        path_is_closed -- set to True if passed paths are closed, False if open

        Returns:
        list of polygons
        """
        cdef cl.Paths solution
        cl.MinkowskiSum(pattern.cl_path(),
                        self.to_clipper_paths(),
                        solution,
                        path_is_closed)
        return PathList.from_clipper_paths(solution)

    cpdef scale_to(self, double s=2**31):
        return PathList([p.scale_to(s) for p in self])

    cpdef scale_from(self, double s=2**31):
        return PathList([p.scale_from(s) for p in self])


cdef class Path:
    cdef:
        np.ndarray _array
        Py_ssize_t _length
        cl.Path    _cl_path

    def __init__(self, object polygon=None):
        if polygon is None:
            self._array = None
            self._length = 0
        else:
            if isinstance(polygon, np.ndarray):
                self._array = polygon.reshape(-1, 2)
            else:
                self._array = np.asarray(polygon, dtype=np.int64).reshape(-1, 2)
            self._length = self._array.size // 2
            self._cl_path = self.to_clipper_path()

    def __repr__(self):
        if self._array is None:
            return "Path(None)"
        return "Path({})".format(self._array.tolist())

    def __getitem__(self, index):
        if self._array is None:
            raise ValueError("{} is not subscriptable".format(self))

        if isinstance(index, slice):
            return Path(self._array[index])
        elif isinstance(index, numbers.Integral):
            if index > self._length:
                raise IndexError("Path index is out of range")
            return self._array[index]
        else:
            raise TypeError("Path indices must be integers")

    def __iter__(self):
        return iter(self._array)

    def __len__(self):
        return self._length

    def __eq__(self, Path other):
        return np.all(self._array == other.array())

    cdef inline cl.Path to_clipper_path(self):
        cdef np.ndarray v
        cdef cl.Path cl_path = cl.Path()
        if self._array is not None:
            for v in self._array:
                cl_path.push_back(cl.IntPoint(v[0], v[1]))
        return cl_path

    @staticmethod
    cdef Path from_clipper_path(cl.Path cl_path, dtype=np.int64):
        cdef Py_ssize_t cl_path_length = cl_path.size()

        cdef np.ndarray py_path_v = np.empty((cl_path_length, 2), dtype=dtype)

        cdef cl.IntPoint point
        cdef Py_ssize_t i
        for i in range(cl_path_length):
            point = cl_path.at(i)
            py_path_v[i][0] = point.X
            py_path_v[i][1] = point.Y

        return Path(py_path_v)

    cpdef np.ndarray array(self):
        return self._array

    cdef inline cl.Path cl_path(self):
        return self._cl_path

    cpdef bint orientation(self):
        return cl.Orientation(self._cl_path)

    cpdef double area(self):
        return cl.Area(self._cl_path)

    cpdef PathList simplify(self, cl.PolyFillType fill_type=cl.pftEvenOdd):
        """ Removes self-intersections from the supplied polygon.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/SimplifyPolygon.htm

        Keyword arguments:
        self -- polygon to be simplified
        fill_type -- PolyFillType used with the boolean union operation

        Returns:
        list of simplified polygons (containing one or more polygons)
        """
        cdef cl.Paths out_polys
        cl.SimplifyPolygon(self._cl_path, out_polys, fill_type)
        return PathList.from_clipper_paths(out_polys)


    cpdef Path clean(self, double distance=1.415):
        """ Removes unnecessary vertices from the provided polygon.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/CleanPolygon.htm

        Keyword arguments:
        poly     -- polygon to be cleaned
        distance -- distance on which vertices are removed, see 'More info' (default: approx. sqrt of 2)

        Returns:
        cleaned polygon
        """
        cdef cl.Path out_poly
        cl.CleanPolygon(self.cl_path(), out_poly, distance)
        return Path.from_clipper_path(out_poly)

    cpdef PathList minkowski_sum(self, Path pattern, bint path_is_closed=True):
        """ Performs Minkowski Addition of the pattern and path.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/MinkowskiSum.htm

        Keyword arguments:
        pattern -- polygon whose points are added to the path
        path    -- open or closed path
        path_is_closed -- set to True if passed path is closed, False if open

        Returns:
        list of polygons (containing one or more polygons)
        """
        cdef cl.Paths solution
        cl.MinkowskiSum(pattern.cl_path(),
                        self.cl_path(),
                        solution,
                        path_is_closed)
        return PathList.from_clipper_paths(solution)

    cpdef scale_to(self, double s=2**31):
        return Path((self._array * s).astype(np.int64))

    cpdef scale_from(self, double s=2**31):
        return Path((self._array / s).astype(np.int64))



cdef class PolyNode:
    """
    Represents ClipperLibs' PolyTree and PolyNode data structures.
    """
    cdef:
        public Path Contour
        public list Childs
        public PolyNode Parent
        public bint IsHole
        public bint IsOpen
        public int depth

    def __init__(self):
        self.Contour = Path()
        self.Childs = []
        self.Parent = None
        self.IsHole = False
        self.IsOpen = False
        self.depth = 0

    cpdef void _filter(self, PathList result, filter_func=None):
        if (filter_func is None or filter_func(self)) and len(self.Contour) > 0:
            result.append(self.Contour)

        for child in self.Childs:
            child._filter(result, filter_func)

    cdef inline bint _is_open(self):
        return self.IsOpen

    cdef inline bint _is_closed(self):
        return not self.IsOpen

    cpdef PathList to_paths(self):
        """ Converts a PolyNode to a list of paths.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/PolyTreeToPaths.htm

        Keyword arguments:
        self -- PolyNode to be filtered

        Returns:
        list of paths
        """
        cdef PathList pathlist = PathList()
        self._filter(pathlist, filter_func=None)
        return pathlist

    cpdef to_paths_closed(self):
        cdef PathList pathlist = PathList()
        self._filter(pathlist, filter_func=self._is_closed)
        return pathlist

    cpdef to_paths_open(self):
        cdef PathList pathlist = PathList()
        self._filter(pathlist, filter_func=self._is_open)
        return pathlist

    @staticmethod
    cdef PolyNode from_clipper_polytree(cl.PolyTree &cl_polytree):
        cdef PolyNode py_polytree = PolyNode()

        cdef list depths = [0]
        cdef cl.PolyNode *cl_child
        cdef PolyNode py_child

        cdef Py_ssize_t i
        for i in range(cl_polytree.ChildCount()):
            cl_child = cl_polytree.Childs[i]
            py_child = PolyNode.node_walk(cl_child, py_polytree)
            py_polytree.Childs.append(py_child)
            depths.append(py_child.depth + 1)
        py_polytree.depth = max(depths)
        return py_polytree

    @staticmethod
    cdef PolyNode node_walk(cl.PolyNode *c_polynode, PolyNode parent):
        cdef PolyNode py_polynode = PolyNode()
        py_polynode.Parent = parent

        py_polynode.IsHole = c_polynode.IsHole()
        py_polynode.IsOpen = c_polynode.IsOpen()

        py_polynode.Contour = Path.from_clipper_path(c_polynode.Contour)

        cdef list depths = [0]
        cdef cl.PolyNode *cl_child
        cdef PolyNode py_child

        cdef Py_ssize_t i
        for i in range(c_polynode.ChildCount()):
            cl_child = c_polynode.Childs[i]
            py_child = PolyNode.node_walk(cl_child, py_polynode)
            depths.append(py_child.depth + 1)
            py_polynode.Childs.append(py_child)

        py_polynode.depth = max(depths)

        return py_polynode


cdef class Clipper:

    """Wraps the Clipper class.

    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/_Body.htm
    """
    cdef cl.Clipper *thisptr  # hold a C++ instance which we're wrapping
    def __cinit__(self):
        """ Creates an instance of the Clipper class. InitOptions from the Clipper class
        are substituted with separate properties.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Methods/Constructor.htm
        """

        self.thisptr = new cl.Clipper()

    def __dealloc__(self):
        del self.thisptr

    cpdef bint add_path(self, Path path, cl.PolyType poly_type, bint closed=True) except *:
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
        cdef cl.Path cl_path = path.cl_path()
        cdef bint result = self.thisptr.AddPath(cl_path, poly_type, closed)
        if not result:
            raise ClipperException('The path is invalid for clipping')
        return result

    cpdef bint add_pathlist(self, PathList path_list, cl.PolyType poly_type, bint closed=True) except *:
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

        cdef cl.Paths cl_paths = path_list.to_clipper_paths()
        cdef bint result = self.thisptr.AddPaths(cl_paths, poly_type, closed)
        if not result:
            raise ClipperException('All paths are invalid for clipping')
        return result

    cpdef void clear(self):
        """ Removes all subject and clip polygons.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperBase/Methods/Clear.htm
        """
        self.thisptr.Clear()

    cpdef Rect get_bounds(self):
        """ Returns an axis-aligned bounding rectangle that bounds all added polygons.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperBase/Methods/GetBounds.htm

        Returns:
        Rect with left, right, bottom, top vertices that define the axis-aligned bounding rectangle.
        """
        return Rect.from_clipper_rect(self.thisptr.GetBounds())

    cpdef PathList execute(self, cl.ClipType clip_type,
                           cl.PolyFillType subj_fill_type=cl.pftEvenOdd,
                           cl.PolyFillType clip_fill_type=cl.pftEvenOdd):
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

        cdef cl.Paths solution
        cdef bint success = self.thisptr.Execute(clip_type, solution, subj_fill_type, clip_fill_type)
        if not success:
            raise ClipperException('Execution of clipper did not succeed!')
        return PathList.from_clipper_paths(solution)

    cpdef PolyNode execute_as_polytree(self, cl.ClipType clip_type,
                 cl.PolyFillType subj_fill_type=cl.pftEvenOdd,
                 cl.PolyFillType clip_fill_type=cl.pftEvenOdd):
        """ Performs the clipping operation and returns a PolyNode.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Methods/Execute.htm

        Keyword arguments:
        clip_type      -- type of the clipping operation
        subj_fill_type -- fill rule of subject paths
        clip_fill_type -- fill rule of clip paths

        Returns:
        PolyNode

        Raises:
        ClipperException -- operation did not succeed
        """
        cdef cl.PolyTree solution
        cdef bint success = self.thisptr.Execute(clip_type, solution, subj_fill_type, clip_fill_type)
        if not success:
            raise ClipperException('Execution of clipper did not succeed!')
        return PolyNode.from_clipper_polytree(solution)

    @property
    def reverse_solution(self):
        """ Should polygons returned from execute/execute_as_polytree have their orientations
        opposite to their normal orientations.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Properties/ReverseSolution.htm
        """
        return self.thisptr.ReverseSolution()

    @reverse_solution.setter
    def reverse_solution(self, bint value):
        self.thisptr.ReverseSolution(value)

    @property
    def preserve_collinear(self):
        """ Should clipper preserve collinear vertices.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Properties/PreserveCollinear.htm
        """
        return self.thisptr.PreserveCollinear()

    @preserve_collinear.setter
    def preserve_collinear(self, bint value):
        self.thisptr.PreserveCollinear(value)

    @property
    def strictly_simple(self):
        """ Should polygons returned from Execute/Execute2 be strictly simple (True) or may be weakly simple (False).
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Properties/StrictlySimple.htm
        """
        return self.thisptr.StrictlySimple()

    @strictly_simple.setter
    def strictly_simple(self, bint value):
        self.thisptr.StrictlySimple(value)


cdef class ClipperOffset:
    """ Wraps the ClipperOffset class.

    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/_Body.htm
    """
    cdef cl.ClipperOffset *thisptr

    def __cinit__(self, double miter_limit=2.0, double arc_tolerance=0.25):
        """ Creates an instance of the ClipperOffset class.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/Constructor.htm
        """
        self.thisptr = new cl.ClipperOffset(miter_limit, arc_tolerance)

    def __dealloc__(self):
        del self.thisptr

    cpdef void add_path(self, Path path, cl.JoinType join_type, cl.EndType end_type):
        """ Add individual path.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/AddPath.htm

        Keyword arguments:
        path      -- path to be added
        join_type -- join type of added path
        end_type  -- end type of added path
        """
        self.thisptr.AddPath(path.cl_path(), join_type, end_type)

    cpdef void add_pathlist(self, PathList pathlist, cl.JoinType join_type, cl.EndType end_type):
        """ Add a list of paths.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/AddPaths.htm

        Keyword arguments:
        path      -- paths to be added
        join_type -- join type of added paths
        end_type  -- end type of added paths
        """
        self.thisptr.AddPaths(pathlist.to_clipper_paths(), join_type, end_type)

    cpdef PathList execute(self, double delta):
        """ Performs the offset operation and returns a list of offset paths.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/Execute.htm

        Keyword arguments:
        delta -- amount to which the supplied paths will be offset - negative delta shrinks polygons,
                 positive delta expands them.

        Returns:
        list of offset paths
        """
        cdef cl.Paths c_solution
        self.thisptr.Execute(c_solution, delta)
        return PathList.from_clipper_paths(c_solution)

    cpdef PolyNode execute_as_polytree(self, double delta):
        """ Performs the offset operation and returns a PolyNode with offset paths.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/Execute.htm

        Keyword arguments:
        delta -- amount to which the supplied paths will be offset - negative delta shrinks polygons,
                 positive delta expands them.

        Returns:
        PolyNode
        """
        cdef cl.PolyTree solution
        self.thisptr.Execute(solution, delta)
        return PolyNode.from_clipper_polytree(solution)

    cpdef void clear(self):
        """ Clears all paths.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/Clear.htm
        """
        self.thisptr.Clear()

    @property
    def miter_limit(self):
        """ Maximum distance in multiples of delta that vertices can be offset from their
        original positions.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Properties/MiterLimit.htm
        """
        return self.thisptr.MiterLimit

    @miter_limit.setter
    def miter_limit(self, double value):
        self.thisptr.MiterLimit = value

    @property
    def arc_tolerance(self):
        """ Maximum acceptable imprecision when arcs are approximated in
        an offsetting operation.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Properties/ArcTolerance.htm
        """

        return self.thisptr.ArcTolerance

    @arc_tolerance.setter
    def arc_tolerance(self, double value):
        self.thisptr.ArcTolerance = value


cpdef Path reverse_path(Path path):
    """ Reverses the vertex order (and hence orientation) in the specified path.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/ReversePath.htm

    Note: Might be more effective to reverse the path outside of this package (eg. via [::-1] on a list)
    so there is no unneeded conversions to internal structures of this package.

    Keyword arguments:
    path -- path to be reversed

    Returns:
    reversed path
    """
    cdef cl.Path cl_path = path.cl_path()
    cl.ReversePath(cl_path)
    return Path.from_clipper_path(cl_path)


cpdef PathList reverse_pathlist(PathList path_list):
    """ Reverses the vertex order (and hence orientation) in the specified path.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/ReversePaths.htm

    Note: Might be more effective to reverse the path outside of this package (eg. via [::-1] on a list)
    so there is no unneeded conversions to internal structures of this package.

    Keyword arguments:
    path_list -- path_list to be reversed

    Returns:
    reversed path_list
    """
    cdef cl.Paths cl_paths = path_list.to_clipper_paths()
    cl.ReversePaths(cl_paths)
    return PathList.from_clipper_paths(cl_paths)


cpdef int point_in_polygon(Point point, Path poly):
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
    return cl.PointInPolygon(point.to_clipper_point(),
                             poly.cl_path())


cpdef PathList minkowski_sum(Path pattern, Path path, bint path_is_closed=True):
    """ Performs Minkowski Addition of the pattern and path.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/MinkowskiSum.htm

    Keyword arguments:
    pattern        -- polygon whose points are added to the path
    path           -- open or closed path
    path_is_closed -- set to True if passed path is closed, False if open

    Returns:
    list of polygons (containing one or more polygons)
    """
    cdef cl.Paths solution
    cl.MinkowskiSum(pattern.to_clipper_path(),
                    path.to_clipper_path(),
                    solution,
                    path_is_closed
    )
    return PathList.from_clipper_paths(solution)


cpdef PathList minkowski_diff(Path poly1, Path poly2):
    """ Performs Minkowski Difference.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/MinkowskiDiff.htm

    Keyword arguments:
    poly1 -- polygon
    poly2 -- polygon

    Returns:
    list of polygons
    """
    cdef cl.Paths solution
    cl.MinkowskiDiff(poly1.to_clipper_path(), poly2.to_clipper_path(), solution)
    return PathList.from_clipper_paths(solution)

