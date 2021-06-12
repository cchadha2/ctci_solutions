import unittest


class ThreeStacks:

    def __init__(self, array_len):
        if array_len < 3:
            raise ValueError("Array must be of length 3 or greater")

        self.arr = [None] * array_len
        self.divisor = array_len // 3
        self.curr_idx = [stack * self.divisor for stack in range(3)]

    def peek(self, stack):
        return self.arr[self.curr_idx[stack]]

    def is_empty(self, stack):
        return ((self.arr[self.curr_idx[stack]] is None)
                 and (self.curr_idx[stack] == stack * self.divisor))

    def push(self, stack, value):
        at_end = (self.curr_idx[stack] == (stack + 1) * self.divisor - 1)
        if at_end and self.arr[self.curr_idx[stack]] is not None:
            raise ValueError("Stack Overflow")

        # If we're anywhere in the middle of the stack, move the index pointer up one for the
        # value to be inserted.
        if not at_end and not self.is_empty(stack):
            self.curr_idx[stack] += 1
        self.arr[self.curr_idx[stack]] = value

    def pop(self, stack):
        if self.is_empty(stack):
            raise ValueError("Can't pop from empty stack")

        return_val, self.arr[self.curr_idx[stack]] = self.arr[self.curr_idx[stack]], None

        if not self.is_empty(stack):
            self.curr_idx[stack] -= 1

        return return_val


class TestThreeStacks(unittest.TestCase):
    """Integration tests for ThreeStacks."""

    def test_lengths(self):
        for length in (15, 20, 3, 100, 1424124):
            divisor = length // 3
            stacks = ThreeStacks(array_len=length)

            stacks.push(0, 15)
            self.assertEqual(stacks.arr[0], 15)

            stacks.push(1, 4)
            self.assertEqual(stacks.arr[divisor], 4)

            stacks.push(2, 9)
            self.assertEqual(stacks.arr[2 * divisor], 9)

            self.assertEqual(stacks.peek(0), 15)

            self.assertEqual(stacks.pop(0), 15)

            self.assertTrue(stacks.is_empty(0))
            self.assertFalse(stacks.is_empty(1))

            with self.assertRaises(ValueError):
                stacks.pop(0)

            for value in range(divisor):
                stacks.push(0, value)

            with self.assertRaises(ValueError):
                stacks.push(0, "I'm gonna fail!")


    def test_empty_stacks(self):
        with self.assertRaises(ValueError):
            ThreeStacks(0)

    def test_length_less_than_three(self):
        with self.assertRaises(ValueError):
            ThreeStacks(2)



if __name__ == "__main__":
    unittest.main()
