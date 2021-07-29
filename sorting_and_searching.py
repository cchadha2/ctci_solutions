import unittest
from dataclasses import dataclass

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


if __name__ == "__main__":
    unittest.main()
