import unittest

from kdtree import KdTree
from kdtree.point import Point


class TestKdTree(unittest.TestCase):

    def test_empty(self):
        tree = KdTree(3, [])

        point = Point([1, 2, 3])
        bounds = [[1, 2], [2, 3], [3, 4]]

        self.assertEqual(tree.get_knn(point, 3), [])
        self.assertEqual(tree.get_nearest(point), None)
        self.assertEqual(tree.get_points_within_bounds(bounds), [])

    def test_empty_add(self):
        tree = KdTree(3, [])

        point = Point([1, 2, 3])
        bounds = [[1, 2], [2, 3], [3, 4]]
        oob_bounds = [[2, 2], [2, 3], [3, 4]]

        tree.add_point(point)

        self.assertEqual(tree.get_knn(point, 3)[0], point)
        self.assertEqual(tree.get_nearest(point), point)
        self.assertEqual(tree.get_points_within_bounds(bounds)[0], point)
        self.assertEqual(tree.get_points_within_bounds(oob_bounds), [])

    def test_single_entry(self):
        point = Point([1, 2, 3])
        tree = KdTree(3, [point])

        bounds = [[1, 2], [2, 3], [3, 4]]
        oob_bounds = [[2, 2], [2, 3], [3, 4]]

        self.assertEqual(tree.get_knn(point, 3)[0], point)
        self.assertEqual(tree.get_nearest(point), point)
        self.assertEqual(tree.get_points_within_bounds(bounds)[0], point)
        self.assertEqual(tree.get_points_within_bounds(oob_bounds), [])

    def test_add_duplicate(self):
        point = Point([1, 2, 3])
        bounds = [[1, 2], [2, 3], [3, 4]]

        tree = KdTree(3, [])
        tree.add_point(point)
        tree.add_point(point)
        tree.add_point(point)

        self.assertEqual(tree.get_points_within_bounds(bounds)[0], point)
        self.assertEqual(len(tree.get_points_within_bounds(bounds)), 3)

    def test_boundary(self):
        bounds = [[1, 3], [1, 2], [3, 4]]
        p1 = Point([1, 2, 3])
        p2 = Point([2, 2, 3])
        p3 = Point([3, 2, 3])
        tree = KdTree(3, [])
        tree.add_point(p1)
        tree.add_point(p2)
        tree.add_point(p3)

        self.assertTrue(p1 in tree.get_points_within_bounds(bounds))
        self.assertTrue(p2 in tree.get_points_within_bounds(bounds))
        self.assertTrue(p3 in tree.get_points_within_bounds(bounds))

        self.assertEqual(len(tree.get_points_within_bounds(bounds)), 3)


if __name__ == '__main__':
    unittest.main()
