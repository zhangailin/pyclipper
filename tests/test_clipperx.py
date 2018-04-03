#!/usr/bin/python
"""
Tests for Pyclipper wrapper library.
"""

from unittest import TestCase, main
import numpy as np
import clipperx

integer_types = (int,)


# Example polygons from http://www.angusj.com/delphi/clipper.php
# square, orientation is False
PATH_SUBJ_1 = [[180, 200], [260, 200], [260, 150], [180, 150]]
PATH_SUBJ_2 = [[215, 160], [230, 190], [200, 190]]  # triangle
PATH_CLIP_1 = [[190, 210], [240, 210], [240, 130], [190, 130]]  # square
# greek letter sigma
PATH_SIGMA = [[300, 400], [100, 400], [200, 300], [100, 200], [300, 200]]
PATTERN = [[4, -6], [6, -6], [-4, 6], [-6, 6]]
INVALID_PATH = [[1, 1], ]  # less than 2 vertices


class TestClipperModule(TestCase):
    def test_has_classes(self):
        self.assertTrue(hasattr(clipperx, 'Point'))
        self.assertTrue(hasattr(clipperx, 'Rect'))
        self.assertTrue(hasattr(clipperx, 'Path'))
        self.assertTrue(hasattr(clipperx, 'PathList'))
        self.assertTrue(hasattr(clipperx, 'PolyNode'))
        self.assertTrue(hasattr(clipperx, 'Clipper'))
        self.assertTrue(hasattr(clipperx, 'ClipperOffset'))

    def test_has_namespace_methods(self):
        for method in ('reverse_path',
                       'reverse_pathlist',
                       'point_in_polygon',
                       'minkowski_sum',
                       'minkowski_diff',
                       ):
            self.assertTrue(hasattr(clipperx, method))


class TestNamespaceMethods(TestCase):
    def setUp(self):
        self.path1 = clipperx.Path(np.array(PATH_SUBJ_1, dtype=np.int64))
        self.path2 = clipperx.Path(np.array(PATH_SUBJ_2, dtype=np.int64))
        self.pattern = clipperx.Path(np.array(PATTERN, dtype=np.int64))
        self.sigma = clipperx.Path(np.array(PATH_SIGMA, dtype=np.int64))

        self.clip1 = clipperx.Path(np.array(PATH_CLIP_1, dtype=np.int64))

    def test_orientation(self):
        self.assertFalse(self.path1.orientation())
        self.assertTrue(self.path1[::-1].orientation())

    def test_area(self):
        # area less than 0 because orientation is False
        area_neg = self.path1.area()
        area_pos = self.path1[::-1].area()
        self.assertLess(area_neg, 0)
        self.assertGreater(area_pos, 0)
        self.assertEqual(abs(area_neg), area_pos)

    def test_point_in_polygon(self):
        # on polygon
        self.assertEqual(clipperx.point_in_polygon(clipperx.Point(180, 200),
                                                   self.path1), -1)

        # in polygon
        self.assertEqual(clipperx.point_in_polygon(clipperx.Point(200, 180),
                                                   self.path1), 1)

        # outside of polygon
        self.assertEqual(clipperx.point_in_polygon(clipperx.Point(500, 500),
                                                   self.path1), 0)

    def test_minkowski_sum(self):
        solution = self.sigma.minkowski_sum(self.pattern, False)
        self.assertGreater(len(solution), 0)

    def test_minkowski_sum2(self):
        sigma_paths = clipperx.PathList([self.sigma])
        solution = sigma_paths.minkowski_sum(self.pattern, False)
        self.assertGreater(len(solution), 0)

    def test_minkowski_diff(self):
        solution = clipperx.minkowski_diff(self.path1, self.path2)
        self.assertGreater(len(solution), 0)

    def test_reverse_path(self):
        solution = clipperx.reverse_path(self.path1)
        manualy_reversed = self.path1[::-1]
        self.check_reversed_path(solution, manualy_reversed)

    def test_reverse_paths(self):
        solution = clipperx.reverse_pathlist(clipperx.PathList([self.path1]))
        manualy_reversed = clipperx.PathList([self.path1[::-1]])
        self.check_reversed_path(solution[0], manualy_reversed[0])

    def check_reversed_path(self, path_1, path_2):
        if len(path_1) is not len(path_2):
            return False

        for i in range(len(path_1)):
            self.assertEqual(path_1[i][0], path_2[i][0])
            self.assertEqual(path_1[i][1], path_2[i][1])

    def test_simplify_polygon(self):
        solution = self.path1.simplify()
        self.assertEqual(len(solution), 1)

    def test_simplify_polygons(self):
        paths = clipperx.PathList([self.path1])
        solution = paths.simplify()
        solution_single = self.path1.simplify()
        self.assertEqual(len(solution), 1)
        self.assertEqual(len(solution), len(solution_single))
        _do_solutions_match(solution, solution_single)

    def test_clean_polygon(self):
        solution = self.clip1.clean()
        self.assertEqual(len(solution), len(self.clip1))

    def test_clean_polygons(self):
        clips = clipperx.PathList([self.clip1])
        solution = clips.clean()
        self.assertEqual(len(solution), 1)
        self.assertEqual(len(solution[0]), len(PATH_CLIP_1))


class TestFilterPolyNode(TestCase):
    def setUp(self):
        clip1 = clipperx.Path(PATH_CLIP_1)
        subj1 = clipperx.Path(PATH_SUBJ_1)
        subj2 = clipperx.Path(PATH_SUBJ_2)
        pattn = clipperx.Path(PATTERN)

        tree = clipperx.PolyNode()
        tree.Contour = clip1
        tree.IsOpen = True

        child = clipperx.PolyNode()
        child.IsOpen = False
        child.Parent = tree
        child.Contour = subj1
        tree.Childs.append(child)

        child = clipperx.PolyNode()
        child.IsOpen = True
        child.Parent = tree
        child.Contour = subj2
        tree.Childs.append(child)

        child2 = clipperx.PolyNode()
        child2.IsOpen = False
        child2.Parent = child
        child2.Contour = pattn
        child.Childs.append(child2)

        # empty contour should not
        # be included in filtered results
        child2 = clipperx.PolyNode()
        child2.IsOpen = False
        child2.Parent = child
        child2.Contour = clipperx.Path([])
        child.Childs.append(child2)

        self.tree = tree

    def test_polytree_to_paths(self):
        paths = self.tree.to_paths()
        self.check_paths(paths, 4)

    def test_closed_paths_from_polytree(self):
        paths = self.tree.to_paths_closed()
        self.check_paths(paths, 2)

    def test_open_paths_from_polytree(self):
        paths = self.tree.to_paths_open()
        self.check_paths(paths, 2)

    def check_paths(self, paths, expected_nr):
        self.assertEqual(len(paths), expected_nr)
        self.assertTrue(all((len(path) > 0 for path in paths)))


class TestClipperAddPaths(TestCase):
    def setUp(self):
        self.pc = clipperx.Clipper()

    def test_add_path(self):
        # should not raise an exception
        self.pc.add_path(clipperx.Path(PATH_CLIP_1),
                         poly_type=clipperx.PT_CLIP)

    def test_add_paths(self):
        # should not raise an exception
        self.pc.add_pathlist(clipperx.PathList([clipperx.Path(PATH_SUBJ_1),
                                                clipperx.Path(PATH_SUBJ_2)]),
                             poly_type=clipperx.PT_SUBJECT)

    def test_add_path_invalid_path(self):
        with self.assertRaises(clipperx.ClipperException):
            self.pc.add_path(clipperx.Path(INVALID_PATH),
                             clipperx.PT_CLIP, True)

    def test_add_paths_invalid_path(self):
        with self.assertRaises(clipperx.ClipperException):
            self.pc.add_pathlist(
                clipperx.PathList([clipperx.Path(INVALID_PATH),
                                   clipperx.Path(INVALID_PATH)]),
                clipperx.PT_CLIP, True)
        try:
            p1 = clipperx.Path(INVALID_PATH)
            p2 = clipperx.Path(PATH_CLIP_1)
            self.pc.add_pathlist(clipperx.PathList([p1, p2]), clipperx.PT_CLIP)
            self.pc.add_pathlist(clipperx.PathList([clipperx.Path(PATH_CLIP_1),
                                                    clipperx.Path(INVALID_PATH)]),
                                 clipperx.PT_CLIP)
        except clipperx.ClipperException:
            self.fail("add_paths raised ClipperException "
                      "when not all paths were invalid")


class TestClassProperties(TestCase):
    def check_property_assignment(self, pc, prop_name, values):
        for val in values:
            setattr(pc, prop_name, val)
            self.assertEqual(getattr(pc, prop_name), val)

    def test_clipper_properties(self):
        pc = clipperx.Clipper()
        for prop_name in ('reverse_solution',
                          'preserve_collinear',
                          'strictly_simple'):
            self.check_property_assignment(pc, prop_name, [True, False])

    def test_clipperoffset_properties(self):
        for factor in range(6):
            pc = clipperx.ClipperOffset()
            for prop_name in ('miter_limit', 'arc_tolerance'):
                self.check_property_assignment(pc, prop_name,
                                               [2.912, 132.12, 12, -123])


class TestClipperExecute(TestCase):
    def setUp(self):
        self.pc = clipperx.Clipper()
        self.add_default_paths(self.pc)
        self.default_args = [clipperx.CT_INTERSECTION,
                             clipperx.PFT_EVENODD,
                             clipperx.PFT_EVENODD]

    @staticmethod
    def add_default_paths(pc):
        pc.add_path(clipperx.Path(PATH_CLIP_1), clipperx.PT_CLIP)
        pc.add_pathlist(clipperx.PathList([
            clipperx.Path(PATH_SUBJ_1),
            clipperx.Path(PATH_SUBJ_2)]), clipperx.PT_SUBJECT)

    @staticmethod
    def add_paths(pc, clip_path, subj_paths, addend=None, multiplier=None):
        pc.add_path(_modify_vertices(clip_path,
                                     addend=addend, multiplier=multiplier),
                    clipperx.PT_CLIP)
        for subj_path in subj_paths:
            pc.add_path(_modify_vertices(subj_path,
                                         addend=addend, multiplier=multiplier),
                        clipperx.PT_SUBJECT)

    def test_get_bounds(self):
        bounds = self.pc.get_bounds()
        self.assertIsInstance(bounds, clipperx.Rect)
        self.assertEqual(bounds.left, 180)
        self.assertEqual(bounds.right, 260)
        self.assertEqual(bounds.top, 130)
        self.assertEqual(bounds.bottom, 210)

    def test_execute(self):
        solution = self.pc.execute(*self.default_args)
        self.assertEqual(len(solution), 2)

    def test_execute_as_polytree(self):
        solution = self.pc.execute_as_polytree(*self.default_args)
        self.assertIsInstance(solution, clipperx.PolyNode)
        self.check_pypolynode(solution)

    def test_execute_empty(self):
        pc = clipperx.Clipper()
        with self.assertRaises(clipperx.ClipperException):
            pc.execute(clipperx.CT_UNION,
                       clipperx.PFT_NONZERO,
                       clipperx.PFT_NONZERO)

    def test_clear(self):
        self.pc.clear()
        with self.assertRaises(clipperx.ClipperException):
            self.pc.execute(*self.default_args)

    def test_exact_results(self):
        """
        Test whether coordinates passed into the library are returned
        exactly, if they are not affected by the operation.
        """

        pc = clipperx.Clipper()

        # Some large triangle.
        path = clipperx.Path([[[0, 1], [0, 0], [15 ** 15, 0]]])

        pc.add_path(path, clipperx.PT_SUBJECT, True)
        result = pc.execute(clipperx.PT_CLIP,
                            clipperx.PFT_EVENODD, clipperx.PFT_EVENODD)

        self.assertEqual(result, clipperx.PathList([path]))

    def check_pypolynode(self, node):
        self.assertTrue(len(node.Contour) is 0 or len(node.Contour) > 2)

        # check vertex coordinate, should not be an iterable (in that case
        # that means that node.Contour is a list of paths, should be path
        if node.Contour:
            self.assertFalse(hasattr(node.Contour[0][0], '__iter__'))

        for child in node.Childs:
            self.check_pypolynode(child)


class TestClipperOffset(TestCase):
    @staticmethod
    def add_path(pc, path):
        pc.add_path(path, clipperx.JT_ROUND, clipperx.ET_CLOSEDPOLYGON)

    def test_execute(self):
        pc = clipperx.ClipperOffset()
        self.add_path(pc, clipperx.Path(PATH_CLIP_1))
        solution = pc.execute(2.0)
        self.assertIsInstance(solution, clipperx.PathList)
        self.assertEqual(len(solution), 1)

    def test_execute_as_polytree(self):
        pc = clipperx.ClipperOffset()
        self.add_path(pc, clipperx.Path(PATH_CLIP_1))
        solution = pc.execute_as_polytree(2.0)
        self.assertIsInstance(solution, clipperx.PolyNode)
        self.assertEqual(len(solution.to_paths_open()), 0)
        self.assertEqual(len(solution.to_paths_closed()), 1)

    def test_clear(self):
        pc = clipperx.ClipperOffset()
        self.add_path(pc, clipperx.Path(PATH_CLIP_1))
        pc.clear()
        solution = pc.execute(2.0)
        self.assertIsInstance(solution, clipperx.PathList)
        self.assertEqual(len(solution), 0)


def _do_solutions_match(paths_1, paths_2, factor=None):
    if len(paths_1) != len(paths_2):
        return False

    paths_1 = [_modify_vertices(p, multiplier=factor, converter=round
                                if factor else None) for p in paths_1]
    paths_2 = [_modify_vertices(p, multiplier=factor, converter=round
                                if factor else None) for p in paths_2]

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

    return clipperx.Path([[convert_coordinate(c) for c in v] for v in path])


def run_tests():
    main()


if __name__ == '__main__':
    run_tests()
