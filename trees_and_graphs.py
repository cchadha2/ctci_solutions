import unittest
from collections import deque
from dataclasses import dataclass, field
from math import exp, inf
from typing import Type


# 4.1
@dataclass
class Node:
    val: object
    children: list = field(default_factory=list)
    marked: bool = False


def path_nodes(source: Type["Node"], target: Type["Node"]):
    queue = deque()
    queue.append(source)
    source.marked = True

    while queue:
        node = queue.popleft()
        for child in node.children:
            if child is target:
                return True
            elif not child.marked:
                queue.append(child)

    return False


class TestPathNodes(unittest.TestCase):

    def test_connected(self):
        target = Node(8)
        root = Node(1, children=[Node(2, children=[Node(4), Node(5), Node(6, children=[Node(7),
            target])]), Node(3, children=[target])])

        self.assertTrue(path_nodes(root, target))

    def test_disconnected(self):
        target = Node(1, children=[Node(2)])
        root = Node(3, children=[Node(4)])

        self.assertFalse(path_nodes(root, target))


# 4.2
@dataclass
class TreeNode:
    val: object
    left: Type["TreeNode"] = None
    right: Type["TreeNode"] = None
    # Used for 4.6
    parent: Type["TreeNode"] = None


def minimal_tree(arr):
    return _minimal_tree(arr, lo=0, hi=len(arr) - 1)


def _minimal_tree(arr, lo, hi):
    if lo > hi:
        return

    mid = (lo + hi) // 2
    node = TreeNode(arr[mid])
    node.left = _minimal_tree(arr, lo, mid - 1)
    node.right = _minimal_tree(arr, mid + 1, hi)

    return node


class TestMinimalTree(unittest.TestCase):

    def test_array(self):
        arr = [1, 4, 7, 11, 24, 98, 101, 600, 804, 1001]

        root = TreeNode(24)
        root.left = TreeNode(4)
        root.left.left = TreeNode(1)
        root.left.right = TreeNode(7)
        root.left.right.right = TreeNode(11)
        root.right = TreeNode(600)
        root.right.right = TreeNode(804)
        root.right.left = TreeNode(98)
        root.right.left.right = TreeNode(101)
        root.right.right.right = TreeNode(1001)

        self.assertEqual(root, minimal_tree(arr))

    def test_none(self):
        self.assertEqual(None, minimal_tree([]))

    def test_small(self):
        arr = [1, 2, 3]

        root = TreeNode(2)
        root.left = TreeNode(1)
        root.right = TreeNode(3)

        self.assertEqual(root, minimal_tree(arr))


# 4.3
@dataclass
class LevelNode:
    val: object
    left: Type["LevelNode"] = None
    right: Type["LevelNode"] = None


@dataclass
class ListNode:
    val: object
    after: Type["ListNode"] = None


def depths(tree: Type["LevelNode"]) -> list[Type["ListNode"]]:
    all_depths = []
    queue = deque()
    queue.append(tree)
    children = []
    list_node = None

    while queue:
        node = queue.popleft()
        if node.left:
            children.append(node.left)
        if node.right:
            children.append(node.right)

        curr = ListNode(node)
        if not list_node:
            list_node = curr
            all_depths.append(list_node)
        else:
            list_node.after = curr
            list_node = list_node.after

        if not queue and children:
            queue.extend(children)
            list_node = None
            children.clear()

    return all_depths


class TestDepths(unittest.TestCase):

    def test_small(self):
        tree = LevelNode(5, left=LevelNode(1, right=LevelNode(8)), right=LevelNode(2))
        expected = [ListNode(tree), ListNode(tree.left, after=ListNode(tree.right)), ListNode(tree.left.right)]

        self.assertEqual(depths(tree), expected)

    def test_bigger(self):
        tree = LevelNode(4,
                left=LevelNode(1,
                    left=LevelNode(6,
                        left=LevelNode(15,
                            left=LevelNode(62), right=LevelNode(47)))),
                right=LevelNode(2,
                    right=LevelNode(7,
                        left=LevelNode(8), right=LevelNode(11))))
        expected = [
                ListNode(tree),
                ListNode(tree.left, after=ListNode(tree.right)),
                ListNode(tree.left.left, after=ListNode(tree.right.right)),
                ListNode(tree.left.left.left, after=ListNode(tree.right.right.left,
                    after=ListNode(tree.right.right.right))),
                ListNode(tree.left.left.left.left, after=ListNode(tree.left.left.left.right))
                ]

        self.assertEqual(depths(tree), expected)

    def test_one(self):
        tree = LevelNode(1)
        expected = [ListNode(tree)]
        self.assertEqual(depths(tree), expected)


# 4.4
def is_balanced(root):
    return find_height(root) != -inf


def find_height(node):
    if not node:
        return -1

    left_height = find_height(node.left)
    if left_height == -inf:
        return left_height

    right_height = find_height(node.right)
    if right_height == -inf:
        return right_height

    return -inf if not abs(left_height - right_height) <= 1 else max(left_height, right_height) + 1


class TestBalanced(unittest.TestCase):

    def test_unbalanced(self):
        root = TreeNode(9, left=TreeNode(12, left=TreeNode(21, left=TreeNode(8),
            right=TreeNode(16))), right=TreeNode(62, right=TreeNode(17)))

        self.assertFalse(is_balanced(root))

    def test_more_balanced(self):
        root = TreeNode(8, left=TreeNode(10, left=TreeNode(12), right=TreeNode(74)),
                right=TreeNode(11, left=TreeNode(6), right=TreeNode(9)))

        self.assertTrue(is_balanced(root))

    def test_another_unbalanced(self):
        root = TreeNode(7, right=TreeNode(4), left=TreeNode(3, left=TreeNode(6, left=TreeNode(8,
            left=TreeNode(21)))))

        self.assertFalse(is_balanced(root))


# 4.5
def validate(node, lo=-inf, hi=inf):
    if not node:
        return True
    elif node.val <= lo or node.val >= hi:
        return False

    return (validate(node.left, lo=lo, hi=node.val) and validate(node.right, lo=node.val, hi=hi))


class TestValidateBST(unittest.TestCase):

    def test_bst(self):
        root = TreeNode(6, left=TreeNode(1, left=TreeNode(-1), right=TreeNode(2)), right=TreeNode(8,
            right=TreeNode(12)))
        self.assertTrue(validate(root))

    def test_bt(self):
        root = TreeNode(7, left=TreeNode(3, left=TreeNode(6), right=TreeNode(18)),
                right=TreeNode(25))
        self.assertFalse(validate(root))

    def test_invalid_bst(self):
        root = TreeNode(5, left=TreeNode(4), right=TreeNode(6, left=TreeNode(3), right=TreeNode(7)))
        self.assertFalse(validate(root))


# 4.6
def successor(node):
    if node.right:
        return _successor(node.right)

    while node.parent and node is node.parent.right:
        node = node.parent

    return node.parent


def _successor(node):
    if not node:
        return

    left = _successor(node.left)
    return left if left else node


class TestSuccessor(unittest.TestCase):

    def test_with_right_subtree(self):
        exp_successor = TreeNode(6)
        root = TreeNode(4, left=TreeNode(2, left=TreeNode(1)), 
                           right=TreeNode(8, right=TreeNode(9), left=exp_successor))
        self.assertEqual(exp_successor, successor(root))

    def test_without_successor(self):
        exp_successor = None
        root = TreeNode(4, left=TreeNode(2, left=TreeNode(1)))
        self.assertEqual(exp_successor, successor(root))


    def test_without_right_subtree(self):
        root = TreeNode(15, left=TreeNode(9, left=TreeNode(6, left=TreeNode(2)),
                            right=TreeNode(12, left=TreeNode(10))), right=TreeNode(18))
        root.left.parent = root
        root.right.parent = root
        root.left.left.parent = root.left
        root.left.right.parent = root.left
        root.left.left.left.parent = root.left.left
        root.left.right.left.parent = root.left.right

        left_predecessor = root.left.left
        self.assertEqual(left_predecessor.parent, successor(left_predecessor))

        right_predecessor = root.left.right
        self.assertEqual(root, successor(right_predecessor))

        maximum = root.right
        self.assertEqual(None, successor(maximum))




if __name__ == "__main__":
    unittest.main()
