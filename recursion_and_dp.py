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



if __name__ == "__main__":
    unittest.main()
