import sys
import unittest
from dataclasses import dataclass
from itertools import zip_longest
from typing import Type


# 10.1
def merge_sorted(a, b):
    a[-len(b):] = b
    merge(a, a.copy(), low=0, middle=len(a) - len(b) - 1, high=len(a) - 1)


def merge(arr, aux, low, middle, high):
    left, right, current = low, middle + 1, low
    while left <= middle and right <= high:
        if aux[left] <= aux[right]:
            arr[current] = aux[left]
            left += 1
        else:
            arr[current] = aux[right]
            right += 1
        current += 1

    for remaining in aux[left : middle + 1]:
        arr[current] = remaining
        current += 1


class TestMerge(unittest.TestCase):

    def test_small(self):
        a = [1, 3, 7, 8, None, None]
        b = [2, 4]

        merge_sorted(a, b)
        self.assertEqual(a, [1, 2, 3, 4, 7, 8])

    def test_bigger(self):
        a = [5, 7, 11, 12, 13, 15, None, None, None]
        b = [4, 9, 14]

        merge_sorted(a, b)
        self.assertEqual(a, [4, 5, 7, 9, 11, 12, 13, 14, 15])

    def test_bigger_with_dupes(self):
        a = [1, 3, 7, 8, 9, 10, 11, 15, 16, 16, 23, None, None, None, None, None, None]
        b = [1, 3, 4, 9, 10, 11]

        merge_sorted(a, b)
        self.assertEqual(a, [1, 1, 3, 3, 4, 7, 8, 9, 9, 10, 10, 11, 11, 15, 16, 16, 23])


# 10.2
@dataclass
class Anagram:
    text: str

    def __gt__(self, other):
        sorted_text = sorted(self.text)
        sorted_other = sorted(other)

        for this, that in zip(sorted_text, sorted_other):
            if this > that:
                return True
            elif this < that:
                return False

        return len(sorted_text) >= len(sorted_other)

    def __iter__(self):
        return iter(self.text)

def anagram_group(words):
    anagrams = {}
    for word in words:
        anagrams.setdefault("".join(sorted(word)), []).append(word)

    words.clear()
    for anagram_groups in anagrams.values():
        words.extend(anagram_groups)


# This option uses the __gt__ method of the anagram class to sort in O(nlogn * slogs) time.
# Where n is number of words in the list and s is the length of the word.
words = [Anagram('ten'), Anagram('net'), Anagram('entuz'), Anagram('flip'), Anagram('sive')]
words.sort()
print(words)

# This option constructs a dictionary in O(n) time and space and uses it to re-populate the original
# list with anagrams grouped together.
words = [Anagram('ten'), Anagram('net'), Anagram('entuz'), Anagram('flip'), Anagram('sive')]
anagram_group(words)
print(words)


# 10.3
def find_num(arr, num, lo, hi):
    mid = (lo + hi) // 2
    if arr[mid] == num:
        return mid

    if arr[mid] > arr[lo]:
        return (find_num(arr, num, lo, mid - 1)
                if arr[lo] <= num < arr[mid]
                else find_num(arr, num, mid + 1, hi))
    elif arr[mid] < arr[hi]:
        return (find_num(arr, num, mid + 1, hi)
                if arr[mid] < num <= arr[hi]
                else find_num(arr, num, lo, mid - 1))
    elif arr[lo] == arr[mid]:
        if arr[mid] != arr[hi]:
            return find_num(arr, num, mid + 1, hi)
        else:
            res = find_num(arr, num, lo, mid - 1)
            return res if res != -1 else find_num(arr, num, mid + 1, hi)

    return -1


arr = [15, 16, 18, 1, 2, 3, 5, 7]
print(find_num(arr, 5, 0, len(arr) - 1))

arr = [1, 1, 1, 1, 1, 2, 3, 5, 7]
print(find_num(arr, 5, 0, len(arr) - 1))


class Listy:

    def __init__(self, *args):
        self.arr = args

    def __getitem__(self, idx):
        return -1 if idx >= len(self.arr) else self.arr[idx]


# 10.4
def find_listy_num(arr, num):
    lo, hi = 0, 1
    if arr[hi] == -1:
        return -1 if arr[0] != num else 0

    while -1 < arr[hi] < num:
        hi *= 2

    return binary_search(arr, num, hi // 2, hi)

def binary_search(arr, num, lo, hi):
    mid = (lo + hi) // 2

    if arr[mid] == num:
        return mid
    # Keep looking to the left if current value is -1.
    elif arr[mid] > num or arr[mid] == -1:
        return binary_search(arr, num, lo, mid - 1)
    elif arr[mid] < num:
        return binary_search(arr, num, mid + 1, hi)

    return -1


arr = Listy(1, 3, 4, 5, 7, 10, 11, 12, 15, 16, 17, 21)
print(arr[50])
idx = find_listy_num(arr, 5)
print(idx, arr[idx])
idx = find_listy_num(arr, 17)
print(idx, arr[idx])


# 10.5
def sparse_search(arr, val, lo, hi):
    if lo > hi:
        return -1

    mid = (lo + hi) // 2
    if arr[mid] == "":
        left = sparse_search(arr, val, lo, mid - 1)
        return left if left != -1 else sparse_search(arr, val, mid + 1, hi)

    if arr[mid] == val:
        return mid
    elif arr[mid] < val:
        return sparse_search(arr, val, mid + 1, hi)
    else:
        return sparse_search(arr, val, lo, mid - 1)


arr = ['at', '', '', '', 'ball', '', '', 'car', '', '']
print(sparse_search(arr, 'ball', 0, len(arr) - 1))
print(sparse_search(arr, 'hi', 0, len(arr) - 1))
print(sparse_search(arr, 'car', 0, len(arr) - 1))
print(sparse_search(arr, 'at', 0, len(arr) - 1))


# 10.6
# This requires an external sort if available memory is < 20GB.
# 1. xGB memory available.
# 2. Sort 20/x chunks individually and save to temp files.
# 3. Load (x / (20 / x) + 1)MB from each chunk (+1 for an output buffer).
# 4. k-way merge from chunks into the output buffer to find min of chunks.
# 5. Write buffer to final sorted file when full.
# 6. Load next (x / (20 / x) + 1)MB from respective temp file when one of the sorted buffers
# empties.
# 7. Repeat until there is no more data in any of the chunks.

# 10.8
class BitVector:

    def __init__(self):
        self.bit_vector = 0

    def get(self, pos):
        return (self.bit_vector & (1 << pos)) != 0

    def set(self, pos):
        self.bit_vector |= (1 << pos)


def check_duplicates(arr):
    print(arr)
    bit_vector = BitVector()
    for elem in arr:
        num = elem - 1 # Numbers start at 1
        if bit_vector.get(num):
            print(f"Duplicate {elem=}")
            continue

        bit_vector.set(num)

arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 12, 13, 14, 15, 16, 17, 18]
check_duplicates(arr)

arr = [1, 5, 1, 10, 12, 10]
check_duplicates(arr)

arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 31999, 2145, 31999,
        890, 7224, 22154, 890]
check_duplicates(arr)


def find_elem(grid, num):
    """O(m + n) time in worst case (e.g. finding inexistent value)."""
    if not grid:
        raise ValueError("Grid cannot be empty")

    row, col = 0, len(grid[0]) - 1
    while row < len(grid) and col >= 0:
        if grid[row][col] == num:
            return row, col
        elif grid[row][col] < num:
            row += 1
        else:
            col -= 1

    return None, None


grid = [[1, 7, 8, 9], [12, 13, 14, 27], [13, 14, 15, 18]]
x, y = find_elem(grid, 15)
print(x, y, grid[x][y])

x, y = find_elem(grid, 7)
print(x, y, grid[x][y])

x, y = find_elem(grid, 2)
print(x, y)


@dataclass
class Node:
    value: object
    left: Type["Node"] = None
    right: Type["Node"] = None
    left_size: int = 0

    def insert(self, val):
        print(f"Currently here {self.value}, with size: {self.left_size} attempting to insert {val}")
        if val <= self.value:
            if not self.left:
                self.left = Node(val)
            else:
                self.left.insert(val)

            self.left_size += 1
        else:
            if not self.right:
                self.right = Node(val)
                return
            else:
                self.right.insert(val)

    def rank(self, val):
        if val == self.value:
            return self.left_size

        if val <= self.value:
            return -1 if not self.left else self.left.rank(val)

        if not self.right:
            return -1
        right = self.right.rank(val)
        return -1 if right == -1 else self.left_size + 1 + right


class BinarySearchTree:

    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = Node(val)
            return

        self.root.insert(val)

    def rank(self, val):
        if not self.root:
            return -1

        return self.root.rank(val)


tree = BinarySearchTree()
for elem in [5, 1, 4, 4, 5, 9, 7, 13, 3]:
    tree.insert(elem)

print(tree.rank(4))
print(tree.rank(3))
print(tree.rank(1))
print(tree.rank(9))
assert(tree.rank(4) == 3)
assert(tree.rank(1) == 0)
assert(tree.rank(3) == 1)
assert(tree.rank(9) == 7)


# 10.11
def alternate_peaks(arr):
    peaks, valleys, remaining = find_extremes(arr)

    aux = arr.copy()
    idx = 0
    first, second = (peaks, valleys) if len(peaks) >= len(valleys) else (valleys, peaks)
    for a, b in zip_longest(first, second):
        arr[idx] = aux[a]
        idx += 1

        if b is not None:
            arr[idx] = aux[b]
            idx += 1

    for remaining_idx in remaining:
        arr[idx] = aux[remaining_idx]
        idx += 1


def find_extremes(arr):
    if len(arr) <= 1:
        raise ValueError("Need more data to find extremes")

    peaks, valleys, remaining = [], [], []

    def first_and_last(arr, idx, adj_idx):
        value = arr[idx]
        if value < arr[adj_idx]:
            valleys.append(idx)
        elif value > arr[adj_idx]:
            peaks.append(idx)
        else:
            remaining.append(idx)
    first_and_last(arr, 0, 1)
    first_and_last(arr, len(arr) - 1, len(arr) - 2)

    for idx, value in enumerate(arr[1:-1], start=1):
        if arr[idx - 1] < value and arr[idx + 1] < value:
            peaks.append(idx)
        elif arr[idx - 1] > value and arr[idx + 1] > value:
            valleys.append(idx)
        else:
            remaining.append(idx)

    return peaks, valleys, remaining

arr = [5, 8, 6, 2, 3, 4, 6]
print(arr)
alternate_peaks(arr)
print(arr)

arr = [5, 3, 1, 2, 3]
print(arr)
alternate_peaks(arr)
print(arr)



if __name__ == "__main__":
    unittest.main()
