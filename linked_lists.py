from python_dsa.linked_list import LinkedList, Node

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


def delete_middle(self, value):
    if self.is_empty:
        return

    node = self._first
    while node and node.after:
        if node.after.value == value and node.after.after:
            node.after = node.after.after
            self._size -= 1
            return

        node = node.after


LinkedList.delete_middle = delete_middle
linked_list = LinkedList(1, 2, 3, 1, 4, 5, 3, 6)
print(linked_list)
linked_list.delete_middle(1)
print(linked_list)
linked_list.delete_middle(6)
print(linked_list)
linked_list.delete_middle(1)
print(linked_list)
linked_list.delete_middle(4)
print(linked_list)
linked_list.delete_middle(3)
print(linked_list)
linked_list.delete_middle(3)
print(linked_list)
linked_list.delete_middle(2)
print(linked_list)
linked_list.delete_middle(5)
print(linked_list)
linked_list.delete_middle(6)
print(linked_list)
linked_list.delete_middle(10)
print(linked_list)
linked_list = LinkedList()
print(linked_list)
linked_list.delete_middle(4)
print(linked_list)


def actual_delete_middle(self, node):
    if not (node and node.after):
        return

    node.value = node.after.value
    node.after = node.after.after


LinkedList.actual_delete_middle = actual_delete_middle
linked_list = LinkedList(1, 2, 3, 4, 5, 6)
node = linked_list._get(3)
linked_list.actual_delete_middle(node)
print(linked_list)
node = linked_list._get(6)
linked_list.actual_delete_middle(node)
print(linked_list)
linked_list = LinkedList(1)
node = linked_list._get(1)
linked_list.actual_delete_middle(node)
print(linked_list)



def partition(self, value):
    new_list = LinkedList()

    node = self._first
    while node:
        if node.value < value:
            new_list.insert_first(node.value)
        else:
            new_list.append(node.value)

        node = node.after

    self._first = new_list._first
    self._last = new_list._last


LinkedList.partition = partition
linked_list = LinkedList(3, 5, 8, 5, 10, 2, 1)
print(linked_list, len(linked_list))
linked_list.partition(5)
print(linked_list, len(linked_list))


def partition(self, value):
    """If the size of the list is already known."""
    node = self._first
    for _ in range(self._size):
        if node.value < value:
            node = node.after
            continue

        # Don't have access to the previous node so we need to delete from the middle of the list as
        # before.
        current_value = node.value
        node.value = node.after.value
        node.after = node.after.after

        # Put deleted value at the end of the list.
        last = Node(current_value)
        self._last.after = last
        self._last = last


LinkedList.partition = partition
linked_list = LinkedList(3, 5, 8, 5, 10, 2, 1)
print(linked_list, len(linked_list))
linked_list.partition(5)
print(linked_list, len(linked_list))
linked_list = LinkedList(10, 3, 6, 5, 10, 2, 1)
print(linked_list, len(linked_list))
linked_list.partition(5)
print(linked_list, len(linked_list))


def partition(self, value):
    """Without creating new Node objects."""
    # Instead of using LinkedList methods, just use the nodes themselves.
    new_head = new_tail = node = self._first

    while node:
        next_node = node.after
        if node.value < value:
            node.after = new_head
            new_head = node
        else:
            new_tail.after = node
            new_tail = node

        node = next_node

    new_tail.after = None
    self._first = new_head
    self._last = new_tail


LinkedList.partition = partition
linked_list = LinkedList(3, 5, 8, 5, 10, 2, 1)
print(linked_list, len(linked_list))
linked_list.partition(5)
print(linked_list, len(linked_list))
linked_list = LinkedList(10, 3, 6, 5, 10, 2, 1)
print(linked_list, len(linked_list))
linked_list.partition(5)
print(linked_list, len(linked_list))


def sum_lists(a, b):
    a_node, b_node = a._first, b._first
    a_sum = b_sum = order = 0
    while a_node or b_node:
        if a_node:
            a_sum += a_node.value * (10 ** order)
            a_node = a_node.after
        if b_node:
            b_sum += b_node.value * (10 ** order)
            b_node = b_node.after

        order += 1

    output = LinkedList()
    output.extend(int(number) for number in reversed(str(a_sum + b_sum)))
    return output


a = LinkedList(7, 1, 6)
b = LinkedList(5, 9, 2)
print(sum_lists(a, b))
a = LinkedList(7, 1, 6, 8, 10)
b = LinkedList(5, 9, 2, 9)
print(sum_lists(a, b))


def sum_lists(a, b):
    """Assuming we don't know linked list length."""
    a_node, b_node = a._first, b._first
    a_sum = []
    b_sum = []

    while a_node:
        a_sum.append(a_node.value)
        a_node = a_node.after

    while b_node:
        b_sum.append(b_node.value)
        b_node = b_node.after

    # print(f"{a_sum=}, {b_sum=}")
    a_sum = int("".join(map(str, a_sum)))
    b_sum = int("".join(map(str, b_sum)))
    # print(f"{a_sum=}, {b_sum=}")


    output = LinkedList()
    output.extend(int(number) for number in str(a_sum + b_sum))
    return output


a = LinkedList(6, 1, 7)
b = LinkedList(2, 9, 5)
print(sum_lists(a, b))
a = LinkedList(10, 8, 6, 1, 7)
b = LinkedList(9, 2, 9, 5)
print(sum_lists(a, b))


# From leetcode.
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    # Typical solution
    dummy_head = ListNode()
    l3 = dummy_head

    carry = 0
    while l1 or l2:
        l1_val = l1.val if l1 else 0
        l2_val = l2.val if l2 else 0
      
        current_sum = l1_val + l2_val + carry
        carry = current_sum // 10
        last_digit = current_sum % 10
      
        new_node = ListNode(last_digit)
        l3.next = new_node
      
        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
        l3 = l3.next
      
    if carry:
        new_node = ListNode(carry)
        l3.next = new_node
        l3 = l3.next
      
    return dummy_head.next


# My new solution.
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    l1_sum = l2_sum = order = 0
    while l1:
        l1_sum += l1.val * (10 ** order)
        l1 = l1.next
        order += 1
        
    order = 0
    while l2:
        l2_sum += l2.val * (10 ** order)
        l2 = l2.next
        order += 1

    sum_iter = reversed(str(l1_sum + l2_sum))
    output = prev_node = ListNode(val=next(sum_iter))
    for number in sum_iter:
        current = ListNode(val=number)
        prev_node.next = current
        prev_node = current
    
    return output


# My original solution.
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
   val1 = ""
   val2 = ""
   
   while (l1):
       val1 += str(l1.val)
       l1 = l1.next
   while (l2):
       val2 += str(l2.val)
       l2 = l2.next
     
   output = str(int(val1[::-1]) + int(val2[::-1]))
   root = l3 = ListNode(int(output[-1]))
   for char in output[-2::-1]:
       l3.next = ListNode(int(char))
       l3 = l3.next
   return root
