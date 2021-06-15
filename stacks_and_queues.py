import unittest
from collections import deque
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


# 3.3
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


# 3.4
class MyQueue:

    def __init__(self):
        self._enqueue = []
        self._dequeue = []

    def enqueue(self, val):
        self._enqueue.append(val)

    def dequeue(self):
        self._shift_vals()
        return self._dequeue.pop()

    def peek(self):
        self._shift_vals()
        return self._dequeue[-1]

    def _shift_vals(self):
        # Worst case time O(n) but average case is O(1).
        if not self._dequeue:
            while self._enqueue:
                self._dequeue.append(self._enqueue.pop())

    def __bool__(self):
        return bool(self._dequeue) or bool(self._enqueue)


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

    def test_peek(self):
        queue = MyQueue()
        queue.enqueue(1)
        self.assertEqual(queue.peek(), 1)
        self.assertTrue(queue)


# 3.5
def sort_stack(stack):
    """In place stack sort."""
    temp = []
    # O(n) time and space.
    while stack:
        temp.append(stack.pop())

    # O(n**2) time in the worst case.
    while temp:
        curr = temp.pop()

        if not stack or stack[-1] >= curr:
            stack.append(curr)
            continue

        while stack and curr > stack[-1]:
            temp.append(stack.pop())
        stack.append(curr)


class TestSortStack(unittest.TestCase):

    def test_small(self):
        stack = [3, 1, 4, 2, 7]
        expected = [7, 4, 3, 2, 1]

        sort_stack(stack)
        self.assertEqual(stack, expected)

    def test_none(self):
        stack = []
        expected = []

        sort_stack(stack)
        self.assertEqual(stack, expected)


    def test_duplicates(self):
        stack = [9, 2, 1, 2, 4, 5]
        expected = [9, 5, 4, 2, 2, 1]

        sort_stack(stack)
        self.assertEqual(stack, expected)


    def test_reverse(self):
        stack = list(range(11))
        expected = list(range(10, -1, -1))

        sort_stack(stack)
        self.assertEqual(stack, expected)


# 3.6
@dataclass
class Animal:
    cat: bool
    order: int = None


class AnimalShelter:

    def __init__(self):
        self.order = 0
        self.cats = deque()
        self.dogs = deque()

    def enqueue(self, animal):
        animal.order = self.order

        if animal.cat:
            self.cats.append(animal)
        else:
            self.dogs.append(animal)

        self.order += 1

    def dequeueAny(self):
        if not self.dogs and not self.cats:
            raise IndexError("Empty shelter!")
        elif not self.dogs:
            return self.cats.popleft()
        elif not self.cats:
            return self.dogs.popleft()

        oldest = self.cats if self.cats[0].order < self.dogs[0].order else self.dogs
        return oldest.popleft()

    def dequeueDog(self):
        if not self.dogs:
            raise IndexError("No dogs in shelter")

        return self.dogs.popleft()

    def dequeueCat(self):
        if not self.cats:
            raise IndexError("No cats in shelter")

        return self.cats.popleft()


class TestAnimalShelter(unittest.TestCase):

    def test_enqueue(self):
        shelter = AnimalShelter()

        fido = Animal(False)
        shelter.enqueue(fido)
        self.assertTrue(fido in shelter.dogs)
        self.assertEqual(fido.order, 0)

        pepper = Animal(True)
        shelter.enqueue(pepper)
        self.assertTrue(pepper in shelter.cats)
        self.assertEqual(pepper.order, 1)

        rex = Animal(False)
        shelter.enqueue(rex)
        self.assertTrue(rex in shelter.dogs)
        self.assertEqual(rex.order, 2)

        cheech = Animal(False)
        shelter.enqueue(cheech)
        self.assertTrue(cheech in shelter.dogs)
        self.assertEqual(cheech.order, 3)

        self.assertEqual(len(shelter.cats), 1)
        self.assertEqual(len(shelter.dogs), 3)

    def test_dequeueAny(self):
        shelter = AnimalShelter()

        fido = Animal(False)
        shelter.enqueue(fido)

        pepper = Animal(True)
        shelter.enqueue(pepper)

        rex = Animal(False)
        shelter.enqueue(rex)

        cheech = Animal(False)
        shelter.enqueue(cheech)

        self.assertEqual(shelter.dequeueAny(), fido)
        self.assertEqual(shelter.dequeueAny(), pepper)
        self.assertEqual(shelter.dequeueAny(), rex)
        self.assertEqual(shelter.dequeueAny(), cheech)

        with self.assertRaises(IndexError):
            shelter.dequeueAny()

    def test_dequeueDog(self):
        shelter = AnimalShelter()

        fido = Animal(False)
        shelter.enqueue(fido)

        rex = Animal(False)
        shelter.enqueue(rex)

        cheech = Animal(False)
        shelter.enqueue(cheech)

        self.assertEqual(shelter.dequeueDog(), fido)
        self.assertEqual(shelter.dequeueDog(), rex)
        self.assertEqual(shelter.dequeueDog(), cheech)

        with self.assertRaises(IndexError):
            shelter.dequeueDog()

    def test_dequeueCat(self):
        shelter = AnimalShelter()

        fido = Animal(True)
        shelter.enqueue(fido)

        rex = Animal(True)
        shelter.enqueue(rex)

        cheech = Animal(True)
        shelter.enqueue(cheech)

        self.assertEqual(shelter.dequeueCat(), fido)
        self.assertEqual(shelter.dequeueCat(), rex)
        self.assertEqual(shelter.dequeueCat(), cheech)

        with self.assertRaises(IndexError):
                shelter.dequeueCat()



if __name__ == "__main__":
    unittest.main()
