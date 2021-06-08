from python_dsa.linked_list import LinkedList

from handy_decorators import timer


# 2.1
@timer
def remove_duplicates(self):
    """Remove duplicates in O(n) time and O(num duplicates) space."""
    node = self._first
    dupes = set()
    dupes.add(node.value)

    while node.after:
        if node.after.value in dupes:
            node.after = node.after.after
            self._size -= 1
            continue

        node = node.after
        dupes.add(node.value)


LinkedList.remove_duplicates = remove_duplicates
linked_list = LinkedList(1, 2, 3, 1, 4, 5, 3, 6)
print(linked_list, len(linked_list))
linked_list.remove_duplicates()
print(linked_list, len(linked_list))
linked_list = LinkedList(1, 2, 3, 1, 4, 5, 3, 6, 8, 1, 2, 4, 6, 5, 1, 5, 0, 2, 9, 1, 2, 4, 5, 1,
                         4, 2, 3, 5, 6, 2, 1, 3, 5, 6, 7)
print(linked_list, len(linked_list))
linked_list.remove_duplicates()
print(linked_list, len(linked_list))


@timer
def remove_duplicates(self):
    """Remove duplicates in O(n**2) time without extra space."""
    node = self._first

    while node.after:
        next_node = node.after
        while next_node.after:
            if next_node.after.value == node.value:
                next_node.after = next_node.after.after
                self._size -= 1
                continue

            next_node = next_node.after

        node = node.after


LinkedList.remove_duplicates = remove_duplicates
linked_list = LinkedList(1, 2, 3, 1, 4, 5, 3, 6)
print(linked_list, len(linked_list))
linked_list.remove_duplicates()
print(linked_list, len(linked_list))
linked_list = LinkedList(1, 2, 3, 1, 4, 5, 3, 6, 8, 1, 2, 4, 6, 5, 1, 5, 0, 2, 9, 1, 2, 4, 5, 1,
                         4, 2, 3, 5, 6, 2, 1, 3, 5, 6, 7)
print(linked_list, len(linked_list))
linked_list.remove_duplicates()
print(linked_list, len(linked_list))


# 2.2
def find_kth_last(self, k):
    if not self.__len__() >= k > 0:
        raise ValueError("k outside of range of linked list")

    kth_node = self._first
    last_node = self._first
    last = 0
    while last_node:
        if last >= k:
            kth_node = kth_node.after

        last_node = last_node.after
        last += 1

    return kth_node


LinkedList.find_kth_last = find_kth_last
print(linked_list, linked_list.find_kth_last(3))
linked_list = LinkedList(1, 2, 3, 1, 4, 5, 3, 6)
print(linked_list, linked_list.find_kth_last(2))


def find_kth_last(self, k):
    """If length of linked list is known."""
    if not self.__len__() >= k > 0:
        raise ValueError("k outside of range of linked list")

    idx = self.__len__() - k
    kth_node = self._first
    for increment in range(idx):
        kth_node = kth_node.after

    return kth_node


LinkedList.find_kth_last = find_kth_last
linked_list = LinkedList(1, 2, 3, 1, 4, 5, 3, 6, 8, 1, 2, 4, 6, 5, 1, 5, 0, 2, 9, 1, 2, 4, 5, 1,
                         4, 2, 3, 5, 6, 2, 1, 3, 5, 6, 7)
linked_list.remove_duplicates()
print(linked_list, linked_list.find_kth_last(3))
linked_list = LinkedList(1, 2, 3, 1, 4, 5, 3, 6)
print(linked_list, linked_list.find_kth_last(2))








