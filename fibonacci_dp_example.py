"""Fibonacci DP examples."""
import unittest

from handy_decorators import timer

@timer
def basic_fib(i: int):
    """Lots of repetitive calls, no memoization."""
    if i <= 1:
        return i

    return basic_fib(i - 2) + basic_fib(i - 1)


@timer
def top_down_fib(i: int):
    """This is an order of magnitude quicker than previous function (O(height) time)."""
    return _top_down_fib(i, [0, 1])

def _top_down_fib(i: int, cache: list):
    if i <= 1:
        return i

    if len(cache) <= i:
        cache.append(_top_down_fib(i - 1, cache) + _top_down_fib(i - 2, cache))

    return cache[i]

@timer
def bottom_up_fib(i: int):
    """Fastest of all due to iterative approach."""
    if i <= 1:
        return i

    prev, curr, counter = 1, 1, 2
    while counter < i:
        next_val = prev + curr
        prev = curr
        curr = next_val
        counter += 1

    return curr


class TestFibonacci(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(basic_fib(4), 3)

    def test_top_down(self):
        self.assertEqual(top_down_fib(4), 3)

    def test_bottom_up(self):
        self.assertEqual(bottom_up_fib(4), 3)


if __name__ == "__main__":
    unittest.main()
