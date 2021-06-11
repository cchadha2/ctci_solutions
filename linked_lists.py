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


# 2.5
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

    def __repr__(self):
        return f"Node(val={self.val}, hash={self.__hash__()})"

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


def palindrome(link):
    a = b = link
    while a:
        b = b.next
        a = a.next
        if a:
            a = a.next

    reverse = None
    while b:
        new_head = b
        b = b.next
        new_head.next = reverse
        reverse = new_head

    a = link
    while reverse:
        if a.val != reverse.val:
            return False

        a, reverse = a.next, reverse.next

    return True


link = ListNode(1, next=ListNode(2, next=ListNode(3, next=ListNode(4, next=ListNode(3,
    next=ListNode(2, next=ListNode(1)))))))
print(palindrome(link))
link = ListNode(1, next=ListNode(2, next=ListNode(2, next=ListNode(1))))
print(palindrome(link))
link = ListNode(1, next=ListNode(4, next=ListNode(6, next=ListNode(8, next=ListNode(10)))))
print(palindrome(link))
link = ListNode(1, next=ListNode(2, next=ListNode(3, next=ListNode(4))))
print(palindrome(link))
link = ListNode(1, next=ListNode(2))
print(palindrome(link))


def intersection(l1, l2):
    l1_node, l2_node = l1, l2
    l1_len = l2_len = 0

    # Find tails.
    while l1_node.next:
        l1_len += 1
        l1_node = l1_node.next
    while l2_node.next:
        l2_len += 1
        l2_node = l2_node.next

    if not l1_node is l2_node:
        return

    # Find parity in length.
    longer = l1 if l1_len >= l2_len else l2
    shorter = l1 if l1_len < l2_len else l2
    diff = abs(l1_len - l2_len)
    for _ in range(diff):
        longer = longer.next

    # Traverse at once.
    while longer is not shorter:
        longer, shorter = longer.next, shorter.next
    return longer


intersecting_node = ListNode(4, next=ListNode(5))
l1 = ListNode(1, next=ListNode(2, next=ListNode(3, next=intersecting_node)))
l2 = ListNode(8, next=intersecting_node)
print(f"Intersection of linked lists: {intersection(l1, l2)}")

intersecting_node = ListNode(6, next=ListNode(9))
l1 = ListNode(1, next=ListNode(2, next=ListNode(3, next=ListNode(4, next=ListNode(5,
    next=intersecting_node)))))
l2 = ListNode(7, next=ListNode(10, next=intersecting_node))
print(f"Intersection of linked lists: {intersection(l1, l2)}")

l1 = ListNode(1, next=ListNode(2, next=ListNode(3, next=ListNode(4, next=ListNode(5)))))
l2 = ListNode(7, next=ListNode(10))
print(f"Intersection of linked lists: {intersection(l1, l2)}")


def loop(l1):
    """Determines whether a loop exists."""
    fast = slow = l1
    while fast and fast.next:
        print(fast.val, slow.val)
        slow = slow.next
        fast = fast.next.next
        if fast is slow:
            return True

    return False


cycle = ListNode(3)
cycle.next = ListNode(4, next=ListNode(5, next=cycle))
l1 = ListNode(1, next=ListNode(2, next=cycle))
res = loop(l1)
print(res)

def detect_cycle(head: ListNode) -> ListNode:
    """Have a fast pointer travel 2 nodes for every 1 that a slow pointer advances.

    Once that fast pointer meets the slow pointer, it'll have travelled twice the distance of the
    slow pointer. Then the distance between the slow pointer and the cycle start node and the head
    and the cycle start node are equal (as shown below). So we advance the head and the slow pointer
    by 1 simultaneously until the two meet (the starting node of the cycle).
    """

    def has_loop(head):
        fast = slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast is slow:
                return slow

    # When fast and slow meet in the cycle, slow has travelled H + D and fast has travelled 2H + 2D.
    # Where H is the distance from head to the starting node of the cycle and D is the distance from
    # the starting node to the crossing point of fast and slow.
    slow = has_loop(head)
    if not slow:
        return

    # When fast meets slow, fast has travelled through the cycle n times => H + D = nL where L is
    # the length of the cycle. Therefore, H (the distance we want to know) = nL - D. And since we
    # are D distance away from the cycle start node currently, we simply need to advance the head
    # at the same rate as the slow pointer until the two meet.
    while head is not slow:
        head = head.next
        slow = slow.next

    return head


cycle = ListNode(3)
cycle.next = ListNode(4, next=ListNode(5, next=cycle))
l1 = ListNode(1, next=ListNode(2, next=cycle))
res = detect_cycle(l1)
print(res)


