class KdTree:

    def __init__(self, dim, points):
        self._dim = dim
        self._points = points
        self._root = self._make_kd_tree()

    # Makes the KD-Tree for fast lookup
    def _make_kd_tree(self):
        if len(self._points) == 0:
            return [None, None, None]
        else:
            return self._make_kd_tree_rec(self._points, self._dim)

    def _make_kd_tree_rec(self, points, dim, i=0):
        if len(points) > 1:
            points.sort(key=lambda x: x[i])
            i = (i + 1) % dim
            half = len(points) >> 1
            return [
                self._make_kd_tree_rec(points[: half], dim, i),
                self._make_kd_tree_rec(points[half + 1:], dim, i),
                points[half]
            ]
        elif len(points) == 1:
            return [None, None, points[0]]

    # Adds a point to the kd-tree\
    def add_point(self, point):
        if self._root[2] is None:
            self._root[2] = point
        else:
            self._add_point_rec(self._root, point)

    def _add_point_rec(self, kd_node, point, i=0):
        if kd_node is not None:
            dx = kd_node[2][i] - point[i]
            i = (i + 1) % self._dim
            for j, c in ((0, dx >= 0), (1, dx < 0)):
                if c and kd_node[j] is None:
                    kd_node[j] = [None, None, point]
                elif c:
                    self._add_point_rec(kd_node[j], point, i)

    def get_knn(self, point, k, dist_func=None, return_distances=False):
        if dist_func is None:
            dist_func = self.dist_sq

        if self._root[2] is None:
            return []

        return self._get_knn_rec(self._root, point, k, dist_func, return_distances)

    # k nearest neighbors
    def _get_knn_rec(self, kd_node, point, k, dist_func, return_distances, i=0, heap=None):
        import heapq
        is_root = not heap
        if is_root:
            heap = []
        if kd_node is not None:
            dist = dist_func(point, kd_node[2])
            dx = kd_node[2][i] - point[i]
            if len(heap) < k:
                heapq.heappush(heap, (-dist, kd_node[2]))
            elif dist < -heap[0][0]:
                heapq.heappushpop(heap, (-dist, kd_node[2]))
            i = (i + 1) % self._dim
            # Goes into the left branch, and then the right branch if needed
            for b in [dx < 0] + [dx >= 0] * (dx * dx < -heap[0][0]):
                self._get_knn_rec(kd_node[b], point, k, dist_func, return_distances, i, heap)
        if is_root:
            neighbors = sorted((-h[0], h[1]) for h in heap)
            return neighbors if return_distances else [n[1] for n in neighbors]

    def get_nearest(self, point, dist_func=None, return_distances=False):
        if dist_func is None:
            dist_func = self.dist_sq

        if self._root[2] is None:
            return None

        return self._get_nearest_rec(self._root, point, dist_func, return_distances)

    # For the closest neighbor
    def _get_nearest_rec(self, kd_node, point, dist_func, return_distances, i=0, best=None):
        if kd_node is not None:
            dist = dist_func(point, kd_node[2])
            dx = kd_node[2][i] - point[i]
            if not best:
                best = [dist, kd_node[2]]
            elif dist < best[0]:
                best[0], best[1] = dist, kd_node[2]
            i = (i + 1) % self._dim
            # Goes into the left branch, and then the right branch if needed
            for b in [dx < 0] + [dx >= 0] * (dx * dx < best[0]):
                self._get_nearest_rec(kd_node[b], point, dist_func, return_distances, i, best)
        return best if return_distances else best[1]

    # For the closest neighbor
    # bounds = [min:max, ..., min:max]

    def get_points_within_bounds(self, bounds):
        if self._root[2] is None:
            return []
        return self.get_points_within_bounds_rec(self._root, bounds)

    def get_points_within_bounds_rec(self, kd_node, bounds, i=0):
        results = []
        if kd_node is not None:

            if self._is_point_within_bounds(kd_node[2], bounds):
                results.append(kd_node[2])

            if kd_node[2][i] > bounds[i][1]:  # go left
                results.extend(self.get_points_within_bounds_rec(kd_node[0], bounds, (i + 1) % self._dim))
            elif kd_node[2][i] < bounds[i][0]:  # go right
                results.extend(self.get_points_within_bounds_rec(kd_node[1], bounds, (i + 1) % self._dim))
            else:  # go both
                results.extend(self.get_points_within_bounds_rec(kd_node[0], bounds, (i + 1) % self._dim))
                results.extend(self.get_points_within_bounds_rec(kd_node[1], bounds, (i + 1) % self._dim))

        return results

    def get_knn_naive(self, point, k, dist_func=None, return_distances=True):
        if dist_func is None:
            dist_func = self.dist_sq

        neighbors = []
        for i, pp in enumerate(self._points):
            dist = dist_func(point, pp)
            neighbors.append((dist, pp))
        neighbors = sorted(neighbors)[:k]
        return neighbors if return_distances else [n[1] for n in neighbors]

    def _is_point_within_bounds(self, point, bounds):

        for i in range(self._dim):
            if point[i] < bounds[i][0] or point[i] > bounds[i][1]:
                return False
        return True

    def dist_sq(self, a, b):
        return sum((a[i] - b[i]) ** 2 for i in range(self._dim))
