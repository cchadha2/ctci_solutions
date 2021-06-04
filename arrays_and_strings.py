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


def palindrome_perm(text):
    """Same complexity but more optimised to avoid 2 iterations over text."""
    # Assume only letters count for palindrome.
    counter = [0] * 26
    text = text.lower() # O(length of text)

    odd_count = 0
    # Only count non whitespace letter characters.
    length = 0
    start_char = "a"
    for char in text: # O(length of text)
        if not char in string.ascii_lowercase:
            continue

        idx = ord(char) - ord(start_char)
        count = counter[idx]
        counter[idx] += 1
        length += 1
        if not count:
            odd_count += 1
            continue

        if not count % 2:
            odd_count += 1
        else:
            odd_count -= 1

    if length % 2 and odd_count == 1:
        return True
    if not length % 2 and not odd_count:
        return True

    return False


print(palindrome_perm("Tact Coa"))
print(palindrome_perm("abcaaab"))
print(palindrome_perm("abcd"))
print(palindrome_perm("aaaa"))


def palindrome_perm(text):
    odds = 0
    length = 0
    text = text.lower()
    for char in text:
        if not char in string.ascii_lowercase:
            continue

        idx = ord(char) - ord("a")
        # This block is method from book.
        mask = 1 << idx
        if not (odds & mask):
            odds |= mask
        else:
            odds &= ~mask
        print(bin(mask), idx, bin(odds))
        # Commented out part here was my original method.
        #if not bin(odds >> idx).endswith('1'):
        #    odds += 2**idx
        #else:
        #    odds -= 2**idx
    print(bin(odds), bin(odds - 1), bin(odds & (odds - 1)))
    return not (odds and bool(odds & (odds - 1)))


print(palindrome_perm("Tact Coa"))
print(palindrome_perm("abcaaab"))
print(palindrome_perm("abcd"))
print(palindrome_perm("aaaa"))


def one_away(word, other):
    if word == other:
        return True
    if abs(len(word) - len(other)) > 1:
        return False

    diffs = i = j = 0
    while i < len(word) or j < len(other):
        print(f"diffs: {diffs}; i: {i}; word: {word}; j: {j}; other: {other}")
        if diffs > 1:
            return False

        if i >= len(word):
            diffs += 1
            j += 1
            continue

        if j >= len(other):
            diffs += 1
            i += 1
            continue

        if word[i] == other[j]:
            i += 1
            j += 1
            continue

        diffs += 1
        if (len(word) - i + 1) > (len(other) - j + 1):
            i += 1
        elif (len(other) - j + 1) > (len(word) - i + 1):
            j += 1
        else:
            i += 1
            j += 1

    print(f"After while loop. diffs: {diffs}; i: {i}; word: {word}; j: {j}; other: {other}")
    return True


print(one_away('pale', 'ple'))
print(one_away('pales', 'pale'))
print(one_away('pale', 'bale'))
print(one_away('pale', 'bake'))
print(one_away('chirag', 'angelica'))
print(one_away('chirag', 'angel'))
print(one_away('chirag', 'angeli'))



