import unittest
from dataclasses import dataclass

# 3.1. Three stacks from one array.
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


# 3.2 Minimum Stack.
@dataclass
class _Node:
    val: object
    stack_min: object


class StackMin:

    def __init__(self):
        self.stack = []
        self.minimum = None

    def push(self, val):
        if not self.stack or val < self.minimum:
            self.minimum = val

        self.stack.append(_Node(val, self.minimum))

    def pop(self):
        return_val = self.stack.pop().val

        if not self.stack:
            self.minimum = None
        elif return_val == self.minimum:
            self.minimum = self.stack[-1].stack_min

        return return_val

    def min(self):
        return self.minimum


class TestStackMin(unittest.TestCase):

    def test_push(self):
        stack = StackMin()
        stack.push(1)

        self.assertEqual(stack.minimum, 1)
        self.assertEqual(stack.stack[0].val, 1)

        stack.push(-1)
        self.assertEqual(stack.minimum, -1)
        self.assertEqual(stack.stack[1].stack_min, -1)
        self.assertEqual(stack.stack[0].stack_min, 1)

    def test_pop(self):
        stack = StackMin()
        for val in (1, 4, -1):
            stack.push(val)
        self.assertEqual(stack.minimum, -1)

        self.assertEqual(stack.pop(), -1)
        self.assertEqual(stack.minimum, 1)

        self.assertEqual(stack.pop(), 4)
        self.assertEqual(stack.minimum, 1)

        self.assertEqual(stack.pop(), 1)
        self.assertIsNone(stack.minimum)

    def test_min(self):
        stack = StackMin()

        self.assertIsNone(stack.minimum)

        stack.push(1)
        self.assertEqual(stack.minimum, 1)

        stack.push(-1)
        self.assertEqual(stack.minimum, -1)


class SetOfStacks:

    capacity = 5

    def __init__(self):
        self.stack_of_stacks = [[None] * self.capacity]
        self.top = 0
        # Will be used for pop_at(idx) method.
        self.curr_idx = [self.top]

    def push(self, val):
        if not (self.top + 1) % self.capacity:
            self.stack_of_stacks.append([None] * self.capacity)
            self.curr_idx.append(0)
            self.top += 1

        row, col = self._1d_to_2d(self.top)
        if (self.top + 1) % self.capacity == 1 and self.stack_of_stacks[row][col] is None:
            self.stack_of_stacks[row][col] = val
            return

        self.top += 1
        self.curr_idx[row] += 1
        row, col = self._1d_to_2d(self.top)
        self.stack_of_stacks[row][col] = val

    def pop(self):
        row, col = self._1d_to_2d(self.top)
        if self.stack_of_stacks[row][col] is None:
            if not self.top:
                raise IndexError("Can't pop from empty stack")

            # Bring top down to next non-None value in stack.
            while self.stack_of_stacks[row][col] is None:
                self.top -= 1
                row, col = self._1d_to_2d(self.top)

        return_val, self.stack_of_stacks[row][col] = self.stack_of_stacks[row][col], None
        if (self.top + 1) % self.capacity == 1:
            if len(self.stack_of_stacks) == 1:
                self.stack_of_stacks[row][col] = None
                return return_val

            self.stack_of_stacks.pop()
            self.curr_idx.pop()
        else:
            self.curr_idx[row] -= 1

        self.top -= 1
        return return_val

    def pop_at(self, row):
        if self.stack_of_stacks[row][0] is None and len(self.stack_of_stacks) == 1:
            raise IndexError("Can't pop from empty stack.")

        col = self.curr_idx[row]
        return_val, self.stack_of_stacks[row][col] = self.stack_of_stacks[row][col], None

        if self.curr_idx[row]:
            self.curr_idx[row] -= 1
        elif len(self.stack_of_stacks) > 1:
            self.curr_idx.pop(row)
            self.stack_of_stacks.pop(row)
            self.top -= self.capacity
        else:
            self.top = 0

        return return_val

    def _1d_to_2d(self, num):
        return num // self.capacity, num % self.capacity


class TestSetOfStacks(unittest.TestCase):

    def test_push(self):
        stack = SetOfStacks()

        stack.push(10)
        self.assertEqual(stack.stack_of_stacks[0][0], 10)
        self.assertEqual(stack.top, 0)

        stack.push(8)
        self.assertEqual(stack.stack_of_stacks[0][1], 8)
        self.assertEqual(stack.top, 1)

        for _ in range(3):
            stack.push(_)
        stack.push(-19)
        self.assertEqual(len(stack.stack_of_stacks), 2)
        self.assertEqual(stack.top, 5)


    def test_pop(self):
        stack = SetOfStacks()

        with self.assertRaises(IndexError):
            stack.pop()

        stack.push(10)
        self.assertEqual(stack.top, 0)
        self.assertEqual(stack.pop(), 10)
        self.assertEqual(stack.top, 0)

        for _ in range(5):
            stack.push(_)
        stack.push(-19)
        self.assertEqual(stack.pop(), -19)
        self.assertEqual(len(stack.stack_of_stacks), 1)
        self.assertEqual(stack.top, 4)

    def test_pop_at(self):
        stack = SetOfStacks()
        for _ in range(7):
            stack.push(_)

        for _ in range(3):
            stack.pop_at(0)

        for _ in range(2):
            stack.pop()

        self.assertEqual(len(stack.stack_of_stacks), 1)
        self.assertFalse(all(stack.stack_of_stacks[0]))

        stack.pop_at(0)
        stack.pop_at(0)
        with self.assertRaises(IndexError):
            stack.pop_at(0)

        for _ in range(7):
            stack.push(_)

        for _ in range(5):
            stack.pop_at(0)

        self.assertEqual(len(stack.stack_of_stacks), 1)
        self.assertEqual(stack.stack_of_stacks[0], [5, 6, None, None, None])


class MyQueue:

    def __init__(self):
        self._enqueue = []
        self._dequeue = []

    def enqueue(self, val):
        self._enqueue.append(val)

    def dequeue(self):
        # Worst case time O(n) but average case is O(1).
        if not self._dequeue:
            while self._enqueue:
                self._dequeue.append(self._enqueue.pop())

        return self._dequeue.pop()

    def __bool__(self):
        return bool(self._dequeue) and bool(self._enqueue)


class TestMyQueue(unittest.TestCase):

    def test_enqueue(self):
        queue = MyQueue()
        queue.enqueue(3)
        self.assertEqual(queue._enqueue[0], 3)
        queue.enqueue(9)
        self.assertEqual(queue._enqueue[1], 9)

    def test_dequeue(self):
        queue = MyQueue()
        for val in range(5):
            queue.enqueue(val)
        for val in range(5):
            self.assertEqual(queue.dequeue(), val)

        self.assertFalse(queue)


if __name__ == "__main__":
    unittest.main()
