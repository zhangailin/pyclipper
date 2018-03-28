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


SILENT = True

"""
SCALING_FACTOR has been deprecated. See https://github.com/greginvm/pyclipper/wiki/Deprecating-SCALING_FACTOR
for an explanation.
"""
SCALING_FACTOR = 1


def log_action(description):
    if not SILENT:
        print(description)

log_action("Python binding clipper library")

import warnings as _warnings



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

cdef cl.Path _to_clipper_path(object polygon):
    cdef cl.Path path = cl.Path()
    for v in polygon:
        path.push_back(cl.IntPoint(v[0], v[1]))
    return path


cdef cl.IntPoint _to_clipper_point(object py_point):
    return cl.IntPoint(py_point[0], py_point[1])


cdef list _from_clipper_paths(cl.Paths paths):
    cdef list polys = []

    cdef cl.Path cl_path
    cdef unsigned int i
    for i in range(paths.size()):
        cl_path = paths.at(i)
        polys.append(Path.from_clipper_path(cl_path))

    return polys


cdef class IntPoint:
    cdef:
        public int64_t X
        public int64_t Y

    def __init__(self, int64_t X, int64_t Y):
        self.X = X
        self.Y = Y

    def __repr__(self):
        return "IntPoint({}, {})".format(self.X, self.Y)

    cdef cl.IntPoint to_clipper_point(self):
        return cl.IntPoint(self.X, self.Y)

    @staticmethod
    cdef IntPoint from_clipper_point(cl.IntPoint cl_point):
        return IntPoint(cl_point.X, cl_point.Y)


cdef class Path:
    cdef:
        np.ndarray _array
        Py_ssize_t _size
        cl.Path _cl_path

    def __init__(self, object polygon=None):
        if polygon is None:
            self._array = None
            self._size = 0
        else:
            self._array = np.asarray(polygon).reshape(-1, 2)
            self._size = self._array.size // 2
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
            if index > self._size:
                raise IndexError("Path index out of range")
            return self._array[index]
        else:
            raise TypeError("Path indices must be integers")

    def __iter__(self):
        return iter(self._array)

    def __len__(self):
        return self._size // 2

    cdef inline cl.Path to_clipper_path(self):
        cdef np.ndarray v
        cdef cl.Path cl_path = cl.Path()
        if self._array is not None:
            for v in self._array:
                cl_path.push_back(cl.IntPoint(v[0], v[1]))
        return cl_path

    @staticmethod
    cdef Path from_clipper_path(cl.Path cl_path, np.dtype dtype=None):
        cdef Py_ssize_t cl_path_size = cl_path.size()
        if dtype is None:
            dtype = np.int64

        cdef np.ndarray py_path_v = np.empty((cl_path_size, 2), dtype=dtype)

        cdef cl.IntPoint point
        cdef Py_ssize_t i
        for i in range(cl_path_size):
            point = cl_path.at(i)
            py_path_v[i][0] = point.X
            py_path_v[i][1] = point.Y

        return Path(py_path_v)

    cdef inline cl.Path cl_path(self):
        return self._cl_path

    cpdef bint orientation(self):
        return cl.Orientation(self._cl_path)

    cpdef double area(self):
        return cl.Area(self._cl_path)

    cpdef list simpify(self, cl.PolyFillType fill_type=cl.pftEvenOdd):
        """ Removes self-intersections from the supplied polygon.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/SimplifyPolygon.htm

        Keyword arguments:
        poly      -- polygon to be simplified
        fill_type -- PolyFillType used with the boolean union operation

        Returns:
        list of simplified polygons (containing one or more polygons)
        """
        cdef cl.Paths out_polys
        cl.SimplifyPolygon(self._cl_path, out_polys, fill_type)
        return _from_clipper_paths(out_polys)


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
        cl.CleanPolygon(self._cl_path, out_poly, distance)
        return Path.from_clipper_path(out_poly)


cpdef int point_in_polygon(IntPoint point, Path poly):
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

#=============================  PolyNode =================
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

#=============================  Other objects ==============

cdef class IntRect:
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
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


class ClipperException(Exception):
    pass

#============================= Namespace functions =========
def Orientation(poly):
    """ Get orientation of the supplied polygon.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/Orientation.htm

    Keyword arguments:
    poly -- closed polygon

    Returns:
    True  -- counter-clockwise orientation
    False -- clockwise orientation
    """
    return <bint>cl.Orientation(_to_clipper_path(poly))


def Area(poly):
    """ Get area of the supplied polygon.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/Area.htm

    Keyword arguments:
    poly -- closed polygon

    Returns:
    Positive number if orientation is True
    Negative number if orientation is False
    """

    return <double>cl.Area(_to_clipper_path(poly))


def PointInPolygon(point, poly):
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

    return <int>cl.PointInPolygon(_to_clipper_point(point),
                                  _to_clipper_path(poly))


def SimplifyPolygon(poly, cl.PolyFillType fill_type=cl.pftEvenOdd):
    """ Removes self-intersections from the supplied polygon.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/SimplifyPolygon.htm

    Keyword arguments:
    poly      -- polygon to be simplified
    fill_type -- PolyFillType used with the boolean union operation

    Returns:
    list of simplified polygons (containing one or more polygons)
    """
    cdef cl.Paths out_polys
    cl.SimplifyPolygon(_to_clipper_path(poly), out_polys, fill_type)
    return _from_clipper_paths(out_polys)


def SimplifyPolygons(polys, cl.PolyFillType fill_type=cl.pftEvenOdd):
    """ Removes self-intersections from the supplied polygons.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/SimplifyPolygons.htm

    Keyword arguments:
    polys     -- polygons to be simplified
    fill_type -- PolyFillType used with the boolean union operation

    Returns:
    list of simplified polygons
    """
    cdef cl.Paths out_polys
    cl.SimplifyPolygons(_to_clipper_paths(polys), out_polys, fill_type)
    return _from_clipper_paths(out_polys)


def CleanPolygon(poly, double distance=1.415):
    """ Removes unnecessary vertices from the provided polygon.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/CleanPolygon.htm

    Keyword arguments:
    poly     -- polygon to be cleaned
    distance -- distance on which vertices are removed, see 'More info' (default: approx. sqrt of 2)

    Returns:
    cleaned polygon
    """
    cdef cl.Path out_poly
    cl.CleanPolygon(_to_clipper_path(poly), out_poly, distance)
    return _from_clipper_path(out_poly)


def CleanPolygons(polys, double distance=1.415):
    """ Removes unnecessary vertices from the provided polygons.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/CleanPolygons.htm

    Keyword arguments:
    polys    -- polygons to be cleaned
    distance -- distance on which vertices are removed, see 'More info' (default: approx. sqrt of 2)

    Returns:
    list of cleaned polygons
    """
    cdef cl.Paths out_polys = _to_clipper_paths(polys)
    cl.CleanPolygons(out_polys, distance)
    return _from_clipper_paths(out_polys)


def MinkowskiSum(pattern, path, bint path_is_closed):
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
    cl.MinkowskiSum(_to_clipper_path(pattern),
                 _to_clipper_path(path),
                 solution,
                 path_is_closed
    )
    return _from_clipper_paths(solution)


def MinkowskiSum2(pattern, paths, bint path_is_closed):
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
    cl.MinkowskiSum(
        _to_clipper_path(pattern),
        _to_clipper_paths(paths),
        solution,
        path_is_closed
    )
    return _from_clipper_paths(solution)


def MinkowskiDiff(poly1, poly2):
    """ Performs Minkowski Difference.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/MinkowskiDiff.htm

    Keyword arguments:
    poly1 -- polygon
    poly2 -- polygon

    Returns:
    list of polygons
    """
    cdef cl.Paths solution
    cl.MinkowskiDiff(_to_clipper_path(poly1), _to_clipper_path(poly2), solution)
    return _from_clipper_paths(solution)


def PolyTreeToPaths(poly_node):
    """ Converts a PolyNode to a list of paths.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/PolyTreeToPaths.htm

    Keyword arguments:
    py_poly_node -- PolyNode to be filtered

    Returns:
    list of paths
    """
    paths = []
    _filter_polynode(poly_node, paths, filter_func=None)
    return paths


def ClosedPathsFromPolyTree(poly_node):
    """ Filters out open paths from the PolyNode and returns only closed paths.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/ClosedPathsFromPolyTree.htm

    Keyword arguments:
    py_poly_node -- PolyNode to be filtered

    Returns:
    list of closed paths
    """

    paths = []
    _filter_polynode(poly_node, paths, filter_func=lambda pn: not pn.IsOpen)
    return paths


def OpenPathsFromPolyTree(PolyNode poly_node):
    """ Filters out closed paths from the PolyNode and returns only open paths.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/OpenPathsFromPolyTree.htm

    Keyword arguments:
    py_poly_node -- PolyNode to be filtered

    Returns:
    list of open paths
    """
    paths = []
    _filter_polynode(poly_node, paths, filter_func=lambda pn: pn.IsOpen)
    return paths


def ReversePath(path):
    """ Reverses the vertex order (and hence orientation) in the specified path.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/ReversePath.htm

    Note: Might be more effective to reverse the path outside of this package (eg. via [::-1] on a list)
    so there is no unneeded conversions to internal structures of this package.

    Keyword arguments:
    path -- path to be reversed

    Returns:
    reversed path
    """
    cdef cl.Path c_path = _to_clipper_path(path)
    cl.ReversePath(c_path)
    return _from_clipper_path(c_path)


def ReversePaths(paths):
    """ Reverses the vertex order (and hence orientation) in each path.
    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Functions/ReversePaths.htm

    Note: Might be more effective to reverse each path outside of this package (eg. via [::-1] on a list)
    so there is no unneeded conversions to internal structures of this package.

    Keyword arguments:
    paths -- paths to be reversed

    Returns:
    list if reversed paths
    """
    cdef cl.Paths c_paths = _to_clipper_paths(paths)
    cl.ReversePaths(c_paths)
    return _from_clipper_paths(c_paths)


def scale_to_clipper(path_or_paths, scale = 2 ** 31):
    """
    Take a path or list of paths with coordinates represented by floats and scale them using the specified factor.
    This function can be user to convert paths to a representation which is more appropriate for Clipper.

    Clipper, and thus Pyclipper, uses 64-bit integers to represent coordinates internally. The actual supported
    range (+/- 2 ** 62) is a bit smaller than the maximal values for this type. To operate on paths which use
    fractional coordinates, it is necessary to translate them from and to a representation which does not depend
    on floats. This can be done using this function and it's reverse, `scale_from_clipper()`.

    For details, see http://www.angusj.com/delphi/clipper/documentation/Docs/Overview/Rounding.htm.

    For example, to perform a clip operation on two polygons, the arguments to `Pyclipper.AddPath()` need to be wrapped
    in `scale_to_clipper()` while the return value needs to be converted back with `scale_from_clipper()`:

    >>> pc = Pyclipper()
    >>> path = [[0, 0], [1, 0], [1 / 2, (3 / 4) ** (1 / 2)]] # A triangle.
    >>> clip = [[0, 1 / 3], [1, 1 / 3], [1, 2 / 3], [0, 1 / 3]] # A rectangle.
    >>> pc.AddPath(scale_to_clipper(path), PT_SUBJECT)
    >>> pc.AddPath(scale_to_clipper(clip), PT_CLIP)
    >>> scale_from_clipper(pc.Execute(CT_INTERSECTION))
    [[[0.6772190444171429, 0.5590730146504939], [0.2383135547861457, 0.41277118446305394],
      [0.19245008938014507, 0.3333333330228925], [0.8075499106198549, 0.3333333330228925]]]

    :param path_or_paths: Either a list of paths or a path. A path is a list of tuples of numbers.
    :param scale: The factor with which to multiply coordinates before converting rounding them to ints. The default
    will give you a range of +/- 2 ** 31 with a precision of 2 ** -31.
    """

    def scale_value(x):
        if hasattr(x, "__len__"):
            return [scale_value(i) for i in x]
        else:
            return <cl.cInt>(<double>x * scale)

    return scale_value(path_or_paths)


def scale_from_clipper(path_or_paths, scale = 2 ** 31):
    """
    Take a path or list of paths with coordinates represented by ints and scale them back to a fractional
    representation. This function does the inverse of `scale_to_clipper()`.

    :param path_or_paths: Either a list of paths or a path. A path is a list of tuples of numbers.
    :param scale: The factor by which to divide coordinates when converting them to floats.
    """

    def scale_value(x):
        if hasattr(x, "__len__"):
            return [scale_value(i) for i in x]
        else:
            return <double>x / scale

    return scale_value(path_or_paths)


cdef class Pyclipper:

    """Wraps the Clipper class.

    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/_Body.htm
    """
    cdef cl.Clipper *thisptr  # hold a C++ instance which we're wrapping
    def __cinit__(self):
        """ Creates an instance of the Clipper class. InitOptions from the Clipper class
        are substituted with separate properties.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Methods/Constructor.htm
        """

        log_action("Creating a Clipper instance")
        self.thisptr = new cl.Clipper()

    def __dealloc__(self):
        log_action("Deleting the Clipper instance")
        del self.thisptr

    def AddPath(self, path, cl.PolyType poly_type, closed=True):
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
        cdef cl.Path c_path = _to_clipper_path(path)
        cdef bint result = <bint> self.thisptr.AddPath(c_path, poly_type, <bint> closed)
        if not result:
            raise ClipperException('The path is invalid for clipping')
        return result

    def AddPaths(self, paths, cl.PolyType poly_type, closed=True):
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

        cdef cl.Paths c_paths = _to_clipper_paths(paths)
        cdef bint result = <bint> self.thisptr.AddPaths(c_paths, poly_type, <bint> closed)
        if not result:
            raise ClipperException('All paths are invalid for clipping')
        return result

    def Clear(self):
        """ Removes all subject and clip polygons.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperBase/Methods/Clear.htm
        """
        self.thisptr.Clear()

    def GetBounds(self):
        """ Returns an axis-aligned bounding rectangle that bounds all added polygons.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperBase/Methods/GetBounds.htm

        Returns:
        IntRect with left, right, bottom, top vertices that define the axis-aligned bounding rectangle.
        """
        _check_scaling_factor()

        cdef cl.IntRect rr = <cl.IntRect> self.thisptr.GetBounds()
        return IntRect(left=rr.left, top=rr.top, right=rr.right, bottom=rr.bottom)

    def Execute(self, cl.ClipType clip_type,
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
        cdef bint success = <bint> self.thisptr.Execute(clip_type, solution, subj_fill_type, clip_fill_type)
        if not success:
            raise ClipperException('Execution of clipper did not succeed!')
        return _from_clipper_paths(solution)

    def Execute2(self, cl.ClipType clip_type,
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
        cdef bint success = <bint> self.thisptr.Execute(clip_type, solution, subj_fill_type, clip_fill_type)
        if not success:
            raise ClipperException('Execution of clipper did not succeed!')
        return _from_poly_tree(solution)

    property ReverseSolution:
        """ Should polygons returned from Execute/Execute2 have their orientations
        opposite to their normal orientations.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Properties/ReverseSolution.htm
        """
        def __get__(self):
            return <bint> self.thisptr.ReverseSolution()

        def __set__(self, value):
            self.thisptr.ReverseSolution(<bint> value)

    property PreserveCollinear:
        """ Should clipper preserve collinear vertices.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Properties/PreserveCollinear.htm
        """
        def __get__(self):
            return <bint> self.thisptr.PreserveCollinear()

        def __set__(self, value):
            self.thisptr.PreserveCollinear(<bint> value)

    property StrictlySimple:
        """ Should polygons returned from Execute/Execute2 be strictly simple (True) or may be weakly simple (False).
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/Clipper/Properties/StrictlySimple.htm
        """
        def __get__(self):
            return <bint> self.thisptr.StrictlySimple()

        def __set__(self, value):
            self.thisptr.StrictlySimple(<bint> value)


cdef class PyclipperOffset:
    """ Wraps the ClipperOffset class.

    More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/_Body.htm
    """
    cdef cl.ClipperOffset *thisptr

    def __cinit__(self, double miter_limit=2.0, double arc_tolerance=0.25):
        """ Creates an instance of the ClipperOffset class.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/Constructor.htm
        """
        log_action("Creating an ClipperOffset instance")
        self.thisptr = new cl.ClipperOffset(miter_limit, arc_tolerance)

    def __dealloc__(self):
        log_action("Deleting the ClipperOffset instance")
        del self.thisptr

    def AddPath(self, path, cl.JoinType join_type, cl.EndType end_type):
        """ Add individual path.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/AddPath.htm

        Keyword arguments:
        path      -- path to be added
        join_type -- join type of added path
        end_type  -- end type of added path
        """
        cdef cl.Path c_path = _to_clipper_path(path)
        self.thisptr.AddPath(c_path, join_type, end_type)

    def AddPaths(self, paths, cl.JoinType join_type, cl.EndType end_type):
        """ Add a list of paths.
        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Methods/AddPaths.htm

        Keyword arguments:
        path      -- paths to be added
        join_type -- join type of added paths
        end_type  -- end type of added paths
        """
        cdef cl.Paths c_paths = _to_clipper_paths(paths)
        self.thisptr.AddPaths(c_paths, join_type, end_type)

    def Execute(self, double delta):
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
        return _from_clipper_paths(c_solution)

    def Execute2(self, double delta):
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
        return _from_poly_tree(solution)

    def Clear(self):
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
            return <double> self.thisptr.MiterLimit

        def __set__(self, value):
            self.thisptr.MiterLimit = <double> value

    property ArcTolerance:
        """ Maximum acceptable imprecision when arcs are approximated in
        an offsetting operation.

        More info: http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Classes/ClipperOffset/Properties/ArcTolerance.htm
        """
        def __get__(self):
            _check_scaling_factor()

            return self.thisptr.ArcTolerance

        def __set__(self, value):
            _check_scaling_factor()

            self.thisptr.ArcTolerance = value


cdef _filter_polynode(pypolynode, result, filter_func=None):
    if (filter_func is None or filter_func(pypolynode)) and len(pypolynode.Contour) > 0:
        result.append(pypolynode.Contour)

    for child in pypolynode.Childs:
        _filter_polynode(child, result, filter_func)


cdef _from_poly_tree(cl.PolyTree &c_polytree):
    cdef PolyNode poly_tree = PolyNode()
    depths = [0]
    for i in xrange(c_polytree.ChildCount()):
        c_child = c_polytree.Childs[i]
        py_child = _node_walk(c_child, poly_tree)
        poly_tree.Childs.append(py_child)
        depths.append(py_child.depth + 1)
    poly_tree.depth = max(depths)
    return poly_tree


cdef _node_walk(cl.PolyNode *c_polynode, object parent):

    cdef PolyNode py_node = PolyNode()
    py_node.Parent = parent

    cdef object ishole = <bint>c_polynode.IsHole()
    py_node.IsHole = ishole

    cdef object isopen = <bint>c_polynode.IsOpen()
    py_node.IsOpen = isopen

    py_node.Contour = _from_clipper_path(c_polynode.Contour)

    # kids
    cdef cl.PolyNode *cNode
    depths = [0]
    for i in range(c_polynode.ChildCount()):
        c_node = c_polynode.Childs[i]
        py_child = _node_walk(c_node, py_node)

        depths.append(py_child.depth + 1)
        py_node.Childs.append(py_child)

    py_node.depth = max(depths)

    return py_node


cdef cl.Paths _to_clipper_paths(object polygons):
    cdef cl.Paths paths = cl.Paths()
    for poly in polygons:
        paths.push_back(_to_clipper_path(poly))
    return paths




cdef object _from_clipper_path(cl.Path path):
    _check_scaling_factor()

    poly = []
    cdef cl.IntPoint point
    for i in range(path.size()):
        point = path[i]
        poly.append([point.X, point.Y])
    return poly


cdef inline _check_scaling_factor():
    """
    Check whether SCALING_FACTOR has been set by the code using this library and warn the user that it has been
    deprecated and it's value is ignored.
    """

    if SCALING_FACTOR != 1:
        _warnings.warn('SCALING_FACTOR is deprecated and it\'s value is ignored. See https://github.com/greginvm/pyclipper/wiki/Deprecating-SCALING_FACTOR for more information.', DeprecationWarning)
