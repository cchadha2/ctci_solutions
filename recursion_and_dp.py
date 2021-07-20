import math
import unittest


# 8.1
def triple_steps(n: int):
    if n < 0:
        raise ValueError("n cannot be a negative number")

    return _triple_steps(n, [0, 1, 2, 4])


def _triple_steps(n: int, cache: list):
    if len(cache) <= n:
        cache.append(_triple_steps(n - 3, cache)
                     + _triple_steps(n - 2, cache)
                     + _triple_steps(n - 1, cache))

    return cache[n]


class TestTripleSteps(unittest.TestCase):

    def test_small_case(self):
        self.assertEqual(triple_steps(5), 13)

    def test_error(self):
        with self.assertRaises(ValueError):
            triple_steps(-1)


# 8.2
class RobotGrid:
    """O(rows x cols) time and O(rows + cols) space complexity."""

    def __init__(self, grid: list):
        if not grid:
            raise ValueError

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.win = self.rows - 1, self.cols - 1

    def robot_path(self):
        return self._robot_path(0, 0)

    def _position_check(self, x, y):
        if (x >= self.rows or y >= self.cols):
            return False

        value = self.grid[x][y]
        if value is None or value == math.inf:
            return False

        return True

    def _robot_path(self, x, y):
        if (x, y) == self.win:
            return True

        down_x, down_y = x + 1, y
        right_x, right_y = x, y + 1
        if self._position_check(right_x, right_y) and self._robot_path(right_x, right_y):
            return True
        if self._position_check(down_x, down_y) and self._robot_path(down_x, down_y):
            return True

        return False


class TestRobotGrid(unittest.TestCase):

    def test_grid(self):
        grid = [[0, 1, None, 3, 4], [5, 6, 7, None, 9], [10, None, 12, None, 14],
                [15, 16, None, None, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
        self.assertTrue(RobotGrid(grid).robot_path())

    def test_error(self):
        with self.assertRaises(ValueError):
            RobotGrid([])

    def test_impossible_grid(self):
        grid = [[0, 1, None, 3, 4], [5, 6, 7, None, 9], [None, None, 12, None, 14],
                [15, 16, None, None, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
        self.assertFalse(RobotGrid(grid).robot_path())


# 8.3
def magic_idx(arr, start, end):
    mid = (start + end) // 2
    if arr[mid] == mid:
        return mid
    if start == end:
        return

    if arr[mid] > mid:
        return magic_idx(arr, start, mid)
    else:
        return magic_idx(arr, mid + 1, end)


def dupe_idx(arr, start, end):
    mid = (start + end) // 2
    if arr[mid] == mid:
        return mid
    if start >= end:
        return

    left, right = min(mid, arr[mid]), max(mid + 1, arr[mid])
    return dupe_idx(arr, start, left) or dupe_idx(arr, right, end)


class TestMagicIdx(unittest.TestCase):

    def test_distinct_left(self):
        arr = [-1, 0, 2, 4, 5, 7, 8, 9, 12]
        self.assertEqual(magic_idx(arr, 0, len(arr)), 2)
        self.assertEqual(dupe_idx(arr, 0, len(arr)), 2)

    def test_distinct_right(self):
        arr = [-12, -4, -1, 0, 1, 5, 8, 12]
        self.assertEqual(magic_idx(arr, 0, len(arr)), 5)
        self.assertEqual(dupe_idx(arr, 0, len(arr)), 5)

    def test_duplicates_left(self):
        arr = [1, 1, 1, 2, 2, 6, 7, 8]
        self.assertIsNone(magic_idx(arr, 0, len(arr)))
        self.assertEqual(dupe_idx(arr, 0, len(arr)), 1)

    def test_duplicates_right(self):
        arr = [-12, 3, 4, 4, 5, 6, 7, 7]
        self.assertIsNone(magic_idx(arr, 0, len(arr)))
        self.assertEqual(dupe_idx(arr, 0, len(arr)), 7)


# 8.4
def power_set(super_set: list):
    return _power_set(super_set, index=0, all_subsets=[])

def _power_set(super_set: list, index: int, all_subsets: list):
    """O(n2**n) time and space complexity in worst case."""
    if len(super_set) == index:
        all_subsets.append([])
    else:
        _power_set(super_set, index + 1, all_subsets)
        item = super_set[index]
        temp_sets = []
        for other_set in all_subsets:
            print(other_set)
            new_set = other_set.copy()
            new_set.append(item)
            temp_sets.append(new_set)
        all_subsets.extend(temp_sets)

    return all_subsets


class TestPowerSet(unittest.TestCase):

    def test_small(self):
        super_set = [1, 2, 3]
        self.assertEqual(power_set(super_set),
                [[], [3], [2], [3, 2], [1], [3, 1], [2, 1], [3, 2, 1]])

if __name__ == "__main__":
    unittest.main()
