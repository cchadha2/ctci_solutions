import string
from collections import Counter


# 1.1
def is_unique(text):
    # Convert to string after calling sorted so
    # that list is garbage collected.
    order = "".join(sorted(text)) # O(nlogn)
    for idx, char in enumerate(order[:-1]): # O(n)
        if char == order[idx + 1]:
            return False

    return True


print(is_unique("nfxv"))
print(is_unique("hhdc"))


# Using an extra data structure.
def is_unique(text):
    # Extended ASCII. O(1) space.
    visited = [False] * 256

    # O(n) time (or O(1) since we cannot have more than 256 unique characters).
    for char in text:
        idx = ord(char)

        if visited[idx]:
            return False

        visited[idx] = True

    return True


print(is_unique("nfxv"))
print(is_unique("hhdc"))


# 1.2
def check_perm(text, other):
    # O(nlogn) time and O(n) space.
    return False if not len(text) == len(other) else sorted(text) == sorted(other)


print(check_perm("hello", "olleh"))
print(check_perm("hello", "hi"))
print(check_perm("hello", "hello!"))


def check_perm(text, other):
    # O(n) time and space.
    return False if not len(text) == len(other) else Counter(text) == Counter(other)


print(check_perm("hello", "olleh"))
print(check_perm("hello", "hi"))
print(check_perm("hello", "hello!"))


# 1.3
def urlify(text, length):
    # Use given true length to index string and replace spaces (O(n) time and space for new string).
    return text[:length].replace(" ", "%20")


print(urlify("Mr John Smith    ", 13))


# 1.4
def palindrome_perm(text):
    """O(length of text) time and O(1) space complexity (under character space assumption)."""
    # Assume only letters count for palindrome.
    counter = [0] * 26
    text = text.lower() # O(length of text)

    start_char = "a"
    for char in text: # O(length of text)
        # Worst case is O(nm) = O(26*1) = O(1) as we'll only ever be checking for 1 character
        # in a string of 26 characters.
        # From link to web archive in source:
        # https://github.com/python/cpython/blob/main/Objects/stringlib/fastsearch.h
        if char in string.ascii_lowercase:
            counter[ord(char) - ord(start_char)] += 1

    # There should only be one non-zero odd count character in a palindrome.
    odd = False
    # O(unique characters in text)
    for count in counter:
        if not count % 2:
            continue

        if odd:
            return False
        else:
            odd = True

    return True


print(palindrome_perm("Tact Coa"))
print(palindrome_perm("abcaaab"))
print(palindrome_perm("abcd"))
print(palindrome_perm("aaaa"))
























