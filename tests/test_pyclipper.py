#!/usr/bin/python
"""
Tests for PyClipper wrapper library.
"""

from __future__ import print_function
from unittest import TestCase, main
import sys
import numpy as np

if sys.version_info < (3,):
    integer_types = (int, long)
else:
    integer_types = (int,)

import PyClipper

# Example polygons from http://www.angusj.com/delphi/clipper.php
PATH_SUBJ_1 = np.array([[180, 200], [260, 200], [260, 150], [180, 150]], dtype=np.int32, order="c")  # square, orientation is False
PATH_SUBJ_2 = np.array([[215, 160], [230, 190], [200, 190]], dtype=np.int32)  # triangle
PATH_CLIP_1 = np.array([[190, 210], [240, 210], [240, 130], [190, 130]], dtype=np.int32)  # square
PATH_SIGMA = np.array([[300, 400], [100, 400], [200, 300], [100, 200], [300, 200]], dtype=np.int32)  # greek letter sigma
PATTERN = np.array([[4, -6], [6, -6], [-4, 6], [-6, 6]], dtype=np.int32)
INVALID_PATH = np.array([[1, 1], ], dtype=np.int32)  # less than 2 vertices


class TestPyClipperModule(TestCase):
    def test_has_classes(self):
        self.assertTrue(hasattr(PyClipper, 'Clipper'))
        self.assertTrue(hasattr(PyClipper, 'ClipperOffset'))

    def test_has_namespace_methods(self):
        for method in ('Orientation', 'Area', 'PointInPolygon', 'SimplifyPolygon', 'SimplifyPolygons',
                       'CleanPolygon', 'CleanPolygons', 'MinkowskiSum', 'MinkowskiSum2', 'MinkowskiDiff',
                       'PolyTreeToPaths', 'ClosedPathsFromPolyTree', 'OpenPathsFromPolyTree',
                       'ReversePath', 'ReversePaths'):
            self.assertTrue(hasattr(PyClipper, method))


class TestNamespaceMethods(TestCase):
    def setUp(self):
        PyClipper.SCALING_FACTOR = 1

    def test_orientation(self):
        self.assertFalse(PyClipper.Orientation(PATH_SUBJ_1))
        self.assertTrue(PyClipper.Orientation(np.ascontiguousarray(PATH_SUBJ_1[::-1])))

    def test_area(self):
        # area less than 0 because orientation is False
        area_neg = PyClipper.Area(PATH_SUBJ_1)
        area_pos = PyClipper.Area(np.ascontiguousarray(PATH_SUBJ_1[::-1]))
        self.assertLess(area_neg, 0)
        self.assertGreater(area_pos, 0)
        self.assertEqual(abs(area_neg), area_pos)

    def test_point_in_polygon(self):
        # on polygon
        self.assertEqual(PyClipper.PointInPolygon(np.array((180, 200), dtype=np.int32), PATH_SUBJ_1), -1)

        # in polygon
        self.assertEqual(PyClipper.PointInPolygon(np.array((200, 180), dtype=np.int32), PATH_SUBJ_1), 1)

        # outside of polygon
        self.assertEqual(PyClipper.PointInPolygon(np.array((500, 500), dtype=np.int32), PATH_SUBJ_1), 0)

    def test_minkowski_sum(self):
        solution = PyClipper.MinkowskiSum(PATTERN, PATH_SIGMA, False)
        self.assertGreater(len(solution), 0)

    def test_minkowski_sum2(self):
        solution = PyClipper.MinkowskiSum2(PATTERN, [PATH_SIGMA], False)
        self.assertGreater(len(solution), 0)

    def test_minkowski_diff(self):
        solution = PyClipper.MinkowskiDiff(PATH_SUBJ_1, PATH_SUBJ_2)
        self.assertGreater(len(solution), 0)

    def test_reverse_path(self):
        solution = PyClipper.ReversePath(PATH_SUBJ_1)
        manualy_reversed = np.ascontiguousarray(PATH_SUBJ_1[::-1])
        print(solution)
        print(manualy_reversed)
        self.check_reversed_path(solution, manualy_reversed)

    def test_reverse_paths(self):
        solution = PyClipper.ReversePaths([PATH_SUBJ_1])
        manualy_reversed = [np.ascontiguousarray(PATH_SUBJ_1[::-1])]
        self.check_reversed_path(solution[0], manualy_reversed[0])

    def check_reversed_path(self, path_1, path_2):
        if len(path_1) is not len(path_2):
            return False

        for i in range(len(path_1)):
            print(path_1[i][0], path_1[i][1])
            #self.assertEqual(path_1[i][0], path_2[i][0])
            #self.assertEqual(path_1[i][1], path_2[i][1])

    def test_simplify_polygon(self):
        solution = PyClipper.SimplifyPolygon(PATH_SUBJ_1)
        self.assertEqual(len(solution), 1)

    def test_simplify_polygons(self):
        solution = PyClipper.SimplifyPolygons([PATH_SUBJ_1])
        solution_single = PyClipper.SimplifyPolygon(PATH_SUBJ_1)
        self.assertEqual(len(solution), 1)
        self.assertEqual(len(solution), len(solution_single))
        _do_solutions_match(solution, solution_single)

    def test_clean_polygon(self):
        solution = PyClipper.CleanPolygon(PATH_CLIP_1)
        self.assertEqual(len(solution), len(PATH_CLIP_1))

    def test_clean_polygons(self):
        solution = PyClipper.CleanPolygons([PATH_CLIP_1])
        self.assertEqual(len(solution), 1)
        self.assertEqual(len(solution[0]), len(PATH_CLIP_1))


class TestFilterPyPolyNode(TestCase):
    def setUp(self):
        tree = PyClipper.PolyNode()
        #tree.Contour.append(PATH_CLIP_1)
        tree.Contour = PATH_CLIP_1
        tree.IsOpen = True

        child = PyClipper.PolyNode()
        child.IsOpen = False
        child.Parent = tree
        child.Contour = PATH_SUBJ_1
        tree.Childs.append(child)

        child = PyClipper.PolyNode()
        child.IsOpen = True
        child.Parent = tree
        child.Contour = PATH_SUBJ_2
        tree.Childs.append(child)

        child2 = PyClipper.PolyNode()
        child2.IsOpen = False
        child2.Parent = child
        child2.Contour = PATTERN
        child.Childs.append(child2)

        # empty contour should not
        # be included in filtered results
        child2 = PyClipper.PolyNode()
        child2.IsOpen = False
        child2.Parent = child
        #child2.Contour = []
        child.Childs.append(child2)

        self.tree = tree

    def test_polytree_to_paths(self):
        paths = PyClipper.PolyTreeToPaths(self.tree)
        self.check_paths(paths, 4)

    def test_closed_paths_from_polytree(self):
        paths = PyClipper.ClosedPathsFromPolyTree(self.tree)
        self.check_paths(paths, 2)

    def test_open_paths_from_polytree(self):
        paths = PyClipper.OpenPathsFromPolyTree(self.tree)
        self.check_paths(paths, 2)

    def check_paths(self, paths, expected_nr):
        self.assertEqual(len(paths), expected_nr)
        self.assertTrue(all((len(path) > 0 for path in paths)))


class TestPyClipperAddPaths(TestCase):
    def setUp(self):
        PyClipper.SCALING_FACTOR = 1
        self.pc = PyClipper.Clipper()

    def test_add_path(self):
        # should not raise an exception
        self.pc.AddPath(PATH_CLIP_1, poly_type=PyClipper.PT_CLIP)

    def test_add_paths(self):
        # should not raise an exception
        self.pc.AddPaths([PATH_SUBJ_1, PATH_SUBJ_2], poly_type=PyClipper.PT_SUBJECT)

    def test_add_path_invalid_path(self):
        self.assertRaises(PyClipper.ClipperException, self.pc.AddPath, INVALID_PATH, PyClipper.PT_CLIP, True)

    def test_add_paths_invalid_path(self):
        self.assertRaises(PyClipper.ClipperException, self.pc.AddPaths, [INVALID_PATH, INVALID_PATH],
                          PyClipper.PT_CLIP, True)
        try:
            self.pc.AddPaths([INVALID_PATH, PATH_CLIP_1], PyClipper.PT_CLIP)
            self.pc.AddPaths([PATH_CLIP_1, INVALID_PATH], PyClipper.PT_CLIP)
        except PyClipper.ClipperException:
            self.fail("add_paths raised ClipperException when not all paths were invalid")


class TestClassProperties(TestCase):
    def check_property_assignment(self, pc, prop_name, values):
        for val in values:
            setattr(pc, prop_name, val)
            self.assertEqual(getattr(pc, prop_name), val)

    def test_PyClipper_properties(self):
        pc = PyClipper.Clipper()
        for prop_name in ('ReverseSolution', 'PreserveCollinear', 'StrictlySimple'):
            self.check_property_assignment(pc, prop_name, [True, False])

    #def test_PyClipperoffset_properties(self):
    #    for factor in range(6):
    #        PyClipper.SCALING_FACTOR = 10 ** factor
    #        pc = PyClipper.PyClipperOffset()
    #        for prop_name in ('MiterLimit', 'ArcTolerance'):
    #            self.check_property_assignment(pc, prop_name, [2.912, 132.12, 12, -123])


class TestPyClipperExecute(TestCase):
    def setUp(self):
        PyClipper.SCALING_FACTOR = 1
        self.pc = PyClipper.Clipper()
        self.add_default_paths(self.pc)
        self.default_args = [PyClipper.CT_INTERSECTION, PyClipper.PFT_EVENODD, PyClipper.PFT_EVENODD]

    @staticmethod
    def add_default_paths(pc):
        pc.AddPath(PATH_CLIP_1, PyClipper.PT_CLIP)
        pc.AddPaths([PATH_SUBJ_1, PATH_SUBJ_2], PyClipper.PT_SUBJECT)

    @staticmethod
    def add_paths(pc, clip_path, subj_paths, addend=None, multiplier=None):
        pc.AddPath(_modify_vertices(clip_path, addend=addend, multiplier=multiplier), PyClipper.PT_CLIP)
        for subj_path in subj_paths:
            pc.AddPath(_modify_vertices(subj_path, addend=addend, multiplier=multiplier), PyClipper.PT_SUBJECT)

    def test_get_bounds(self):
        bounds = self.pc.GetBounds()
        self.assertIsInstance(bounds, PyClipper.IntRect)
        self.assertEqual(bounds.left, 180)
        self.assertEqual(bounds.right, 260)
        self.assertEqual(bounds.top, 130)
        self.assertEqual(bounds.bottom, 210)

    def test_execute(self):
        solution = self.pc.Execute(*self.default_args)
        self.assertEqual(len(solution), 2)

    def test_execute2(self):
        solution = self.pc.Execute2(*self.default_args)
        self.assertIsInstance(solution, PyClipper.PolyNode)
        self.check_pypolynode(solution)

    def test_execute_empty(self):
        pc = PyClipper.Clipper()
        with self.assertRaises(PyClipper.ClipperException):
            pc.Execute(PyClipper.CT_UNION,
                       PyClipper.PFT_NONZERO,
                       PyClipper.PFT_NONZERO)

    def test_clear(self):
        self.pc.Clear()
        with self.assertRaises(PyClipper.ClipperException):
            self.pc.Execute(*self.default_args)

    def test_exact_results(self):
        """
        Test whether coordinates passed into the library are returned exactly, if they are not affected by the
        operation.
        """

        pc = PyClipper.Clipper()

        # Some large triangle.
        path = np.array([[0, 1], [0, 0], [2 ** 31 - 7, 0]], dtype=np.int32)

        pc.AddPaths([path], PyClipper.PT_SUBJECT, True)
        result = pc.Execute(PyClipper.PT_CLIP, PyClipper.PFT_EVENODD, PyClipper.PFT_EVENODD)

        assert result == path

    def check_pypolynode(self, node):
        self.assertTrue(len(node.Contour) is 0 or len(node.Contour) > 2)

        # check vertex coordinate, should not be an iterable (in that case
        # that means that node.Contour is a list of paths, should be path
        if node.Contour:
            self.assertFalse(hasattr(node.Contour[0][0], '__iter__'))

        for child in node.Childs:
            self.check_pypolynode(child)


class TestPyClipperOffset(TestCase):
    def setUp(self):
        PyClipper.SCALING_FACTOR = 1

    @staticmethod
    def add_path(pc, path):
        pc.AddPath(path, PyClipper.JT_ROUND, PyClipper.ET_CLOSEDPOLYGON)

    def test_execute(self):
        pc = PyClipper.ClipperOffset()
        self.add_path(pc, PATH_CLIP_1)
        solution = pc.Execute(2.0)
        self.assertIsInstance(solution, list)
        self.assertEqual(len(solution), 1)

    def test_execute2(self):
        pc = PyClipper.ClipperOffset()
        self.add_path(pc, PATH_CLIP_1)
        solution = pc.Execute2(2.0)
        self.assertIsInstance(solution, PyClipper.PolyNode)
        self.assertEqual(len(PyClipper.OpenPathsFromPolyTree(solution)), 0)
        self.assertEqual(len(PyClipper.ClosedPathsFromPolyTree(solution)), 1)

    def test_clear(self):
        pc = PyClipper.ClipperOffset()
        self.add_path(pc, PATH_CLIP_1)
        pc.Clear()
        solution = pc.Execute(2.0)
        self.assertIsInstance(solution, list)
        self.assertEqual(len(solution), 0)

'''
class TestScalingFactorWarning(TestCase):
    def setUp(self):
        PyClipper.SCALING_FACTOR = 2.
        self.pc = PyClipper.PyClipper()

    def test_orientation(self):
        with self.assertWarns(DeprecationWarning):
            PyClipper.Orientation(PATH_SUBJ_1)

    def test_area(self):
        with self.assertWarns(DeprecationWarning):
            PyClipper.Area(PATH_SUBJ_1)

    def test_point_in_polygon(self):
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(PyClipper.PointInPolygon((180, 200), PATH_SUBJ_1), -1)

    def test_minkowski_sum(self):
        with self.assertWarns(DeprecationWarning):
            PyClipper.MinkowskiSum(PATTERN, PATH_SIGMA, False)

    def test_minkowski_sum2(self):
        with self.assertWarns(DeprecationWarning):
            PyClipper.MinkowskiSum2(PATTERN, [PATH_SIGMA], False)

    def test_minkowski_diff(self):
        with self.assertWarns(DeprecationWarning):
            PyClipper.MinkowskiDiff(PATH_SUBJ_1, PATH_SUBJ_2)

    def test_add_path(self):
        with self.assertWarns(DeprecationWarning):
            self.pc.AddPath(PATH_CLIP_1, poly_type=PyClipper.PT_CLIP)

    def test_add_paths(self):
        with self.assertWarns(DeprecationWarning):
            self.pc.AddPaths([PATH_SUBJ_1, PATH_SUBJ_2], poly_type=PyClipper.PT_SUBJECT)
'''
'''
class TestScalingFunctions(TestCase):
    scale = 2 ** 31
    path = [(0, 0), (1, 1)]
    paths = [path] * 3

    def test_value_scale_to(self):
        value = 0.5
        res = PyClipper.scale_to_clipper(value, self.scale)

        assert isinstance(res, integer_types)
        assert res == int(value * self.scale)

    def test_value_scale_from(self):
        value = 1000000000000
        res = PyClipper.scale_from_clipper(value, self.scale)

        assert isinstance(res, float)
        # Convert to float to get "normal" division in Python < 3.
        assert res == float(value) / self.scale

    def test_path_scale_to(self):
        res = PyClipper.scale_to_clipper(self.path)

        assert len(res) == len(self.path)
        assert all(isinstance(i, list) for i in res)
        assert all(isinstance(j, integer_types) for i in res for j in i)

    def test_path_scale_from(self):
        res = PyClipper.scale_from_clipper(self.path)

        assert len(res) == len(self.path)
        assert all(isinstance(i, list) for i in res)
        assert all(isinstance(j, float) for i in res for j in i)

    def test_paths_scale_to(self):
        res = PyClipper.scale_to_clipper(self.paths)

        assert len(res) == len(self.paths)
        assert all(isinstance(i, list) for i in res)
        assert all(isinstance(j, list) for i in res for j in i)
        assert all(isinstance(k, integer_types) for i in res for j in i for k in j)

    def test_paths_scale_from(self):
        res = PyClipper.scale_from_clipper(self.paths)

        assert len(res) == len(self.paths)
        assert all(isinstance(i, list) for i in res)
        assert all(isinstance(j, list) for i in res for j in i)
        assert all(isinstance(k, float) for i in res for j in i for k in j)
'''

'''
class TestNonStandardNumbers(TestCase):

    def test_sympyzero(self):
        try:
            from sympy import Point2D
            from sympy.core.numbers import Zero
        except ImportError:
            self.skipTest("Skipping, sympy not available")

        path = [(0,0), (0,1)]
        path = [Point2D(v) for v in [(0,0), (0,1)]]
        assert type(path[0].x) == Zero
        path = PyClipper.scale_to_clipper(path)
        assert path == [[0, 0], [0, 2147483648]]
'''


def _do_solutions_match(paths_1, paths_2, factor=None):
    if len(paths_1) != len(paths_2):
        return False

    paths_1 = [_modify_vertices(p, multiplier=factor, converter=round if factor else None) for p in paths_1]
    paths_2 = [_modify_vertices(p, multiplier=factor, converter=round if factor else None) for p in paths_2]

    return all(((p_1 in paths_2) for p_1 in paths_1))


def _modify_vertices(path, addend=0.0, multiplier=1.0, converter=None):
    path = path[:]

    def convert_coordinate(c):
        if multiplier is not None:
            c *= multiplier
        if addend is not None:
            c += addend
        if converter:
            c = converter(c)
        return c

    return [[convert_coordinate(c) for c in v] for v in path]


def run_tests():
    main()


if __name__ == '__main__':
    run_tests()
