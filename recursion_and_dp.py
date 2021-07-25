import itertools
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
            new_set = other_set.copy()
            new_set.append(item)
            temp_sets.append(new_set)
        all_subsets.extend(temp_sets)

    return all_subsets


def pythonic_power_set(super_set):
    """Same time and space complexity but a bit more pythonic."""
    return [list(elem) for r in range(len(super_set) + 1)
            for elem in itertools.combinations(super_set, r)]


def more_pythonic_power_set(super_set):
    """Lazy evaluation and yields tuples instead of lists or sets."""
    return itertools.chain_from_iterable(itertools.combinations(super_set, r)
            for r in range(len(super_set) + 1))


class TestPowerSet(unittest.TestCase):

    def test_small(self):
        super_set = [1, 2, 3]
        self.assertEqual(power_set(super_set),
                [[], [3], [2], [3, 2], [1], [3, 1], [2, 1], [3, 2, 1]])
        self.assertEqual(pythonic_power_set(super_set),
                [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]])


# 8.5
def multiply(a, b):
    bigger, smaller = (a, b) if a >= b else (b, a)
    print(f"Start with {bigger}, {smaller}")
    return _multiply(smaller, bigger)

def _multiply(smaller, bigger):
    print(f"Now at {bigger}, {smaller}")
    if not smaller:
        return 0
    elif smaller == 1:
        return bigger

    smaller_b = smaller >> 1
    half_prod = _multiply(smaller_b, bigger)
    print(f"Current half: {half_prod}")

    result = half_prod + half_prod
    print(f"Result if smaller is even: {result}, else: {result + bigger}")
    return result + bigger if smaller % 2 else result

class TestRecursiveMultiply(unittest.TestCase):

    def test_evens(self):
        self.assertEqual(multiply(8, 9), 72)
        self.assertEqual(multiply(4, 9), 36)
        self.assertEqual(multiply(5, 5), 25)


def towers(a, b, c, n):
    if n == 1:
        c.append(a.pop())
        return

    # Move n - 1 disks to b (temporarily).
    towers(a, c, b, n - 1)
    # Base case - move nth disk from a to c (end).
    towers(a, b, c, 1)
    # Move n - 1 disks from b to c (using a as temp).
    towers(b, a, c, n - 1)


class TestTowersOfHanoi(unittest.TestCase):

    def test_three(self):
        a = [3, 2, 1]
        b = []
        c = []

        towers(a, b, c, n=3)
        print("Finished calls", end="\n")
        self.assertEqual(c, [3, 2, 1])

    def test_two(self):
        a = [2, 1]
        b = []
        c = []

        towers(a, b, c, n=2)
        print("Finished calls", end="\n")
        self.assertEqual(c, [2, 1])

    def test_one(self):
        a = [1]
        b = []
        c = []

        towers(a, b, c, n=1)
        print("Finished calls", end="\n")
        self.assertEqual(c, [1])


    def test_five(self):
        a = [5, 4, 3, 2, 1]
        b = []
        c = []

        towers(a, b, c, n=5)
        print("Finished calls", end="\n")
        self.assertEqual(c, [5, 4, 3, 2, 1])

    def test_ten(self):
        a = list(range(1, 11))
        b = []
        c = []

        towers(a, b, c, n=10)
        self.assertEqual(c, list(range(1, 11)))


# 8.7
def perms(s, start, end):
    if (end - start) == 1:
        return [s[start : end]]

    # O(n) time and space
    base_texts = perms(s, start, end - 1)
    new_char = s[end - 1 : end]

    # Worst case time = O(n**2 * n!). Worse case space = O(n!)
    res = []
    # O(n!) time and space for while loop.
    while base_texts:
        base = base_texts.pop()
        res.append(new_char + base)
        # O((n - 1) * n) = O(n**2)
        res.extend(base[idx:] + new_char + base[:idx] for idx in range(len(base)))

    return res


class TestPermutations(unittest.TestCase):

    def test_small(self):
        text = "ABC"
        res = perms(text, 0, len(text))
        self.assertEqual(['CAB', 'ABC', 'BCA', 'CBA', 'BAC', 'ACB'], res)

    def test_bigger(self):
        text = "ABCD"
        res = perms(text, 0, len(text))
        self.assertEqual(4 * 3 * 2, len(res))

# 8.8
def perm_with_dupes(s, start, end):
    """Same runtime as previous function."""
    if (end - start) == 1:
        return set(s[start : end])

    # O(n) time and space
    base_texts = perms(s, start, end - 1)
    new_char = s[end - 1 : end]

    # Worst case time = O(n**2 * n!). Worse case space = O(n!)
    res = set()
    # O(n!) time and space for while loop.
    while base_texts:
        base = base_texts.pop()
        res.add(new_char + base)
        # O((n - 1) * n) = O(n**2)
        res.update(base[idx:] + new_char + base[:idx] for idx in range(len(base)))

    return res


class TestDupePermutations(unittest.TestCase):

    def test_small(self):
        text = "AAB"
        res = perm_with_dupes(text, 0, len(text))
        self.assertEqual(3, len(res))

    def test_bigger(self):
        text = "AABCC"
        res = perm_with_dupes(text, 0, len(text))
        self.assertEqual((5 * 4 * 3 * 2) / (2 * 2), len(res))

    def test_all_dupes(self):
        # Takes a very long time.
        text = "AAAAAAAAA"
        res = perm_with_dupes(text, 0, len(text))
        self.assertEqual(1, len(res))


# 8.9
def add_parens(res, left_rems, right_rems, text, idx):
    if left_rems < 0 or right_rems < left_rems:
        return

    if not left_rems and not right_rems:
        res.append("".join(text))
        return

    # Add a left parenthesis.
    text[idx] = "("
    add_parens(res, left_rems - 1, right_rems, text, idx + 1)

    # Add a right parenthesis.
    text[idx] = ")"
    add_parens(res, left_rems, right_rems - 1, text, idx + 1)


def parens(count):
    res = []
    # Total number parentheses in a value is count * 2.
    text = [None] * (count * 2)

    add_parens(res, left_rems=count, right_rems=count, text=text, idx=0)

    return res


class TestParens(unittest.TestCase):

    def test_one(self):
        exp = ["()"]
        self.assertEqual(parens(1), exp)

    def test_two(self):
        exp = ["(())", "()()"]
        self.assertEqual(parens(2), exp)

    def test_three(self):
        exp = ["((()))", "(()())", "(())()", "()(())", "()()()"]
        self.assertEqual(parens(3), exp)




if __name__ == "__main__":
    unittest.main()
