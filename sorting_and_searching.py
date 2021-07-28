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



if __name__ == "__main__":
    unittest.main()
