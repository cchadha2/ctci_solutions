import random
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


# 4.7
# Assuming projects are alphabetical letters, in sequence, represented as strings.
class DiGraph:
    def __init__(self, vertices, edges):
        self.initial = ord("a")
        # Adjacency lists representation of graph, assuming more vertices may be later added.
        self.graph = [[] for _ in vertices]
        for from_vertex, to_vertex in edges:
            self.graph[ord(from_vertex) - self.initial].append(to_vertex)

    def cycle_finder(self):
        """O(E + V) as every vertex and every edge are visited at most once."""
        path = set()
        visited = set()

        if any(self._visit(chr(vertex_idx + self.initial), visited, path)
               for vertex_idx, _ in enumerate(self.graph)):
            raise ValueError("Graph is not a directed acyclic graph")


    def topological_sort(self):
        """O(E + V) as every vertex and every edge are visited at most once."""
        self.cycle_finder()

        post_order = []
        visited = set()
        for vertex_idx, _ in enumerate(self.graph):
            vertex = chr(vertex_idx + self.initial)
            self._topological_sort(vertex, visited, post_order)

        return reversed(post_order)

    def _visit(self, vertex, visited, path):
        if vertex in visited:
            return False

        visited.add(vertex)
        path.add(vertex)
        for neighbour in self.graph[ord(vertex) - self.initial]:
            if neighbour in path or self._visit(neighbour, visited, path):
                return True
        path.remove(vertex)

        return False

    def _topological_sort(self, vertex, visited, post_order):
        if vertex in visited:
            return

        visited.add(vertex)
        for neighbour in self.graph[ord(vertex) - self.initial]:
            self._topological_sort(neighbour, visited, post_order)

        post_order.append(vertex)


class TestTopologicalSort(unittest.TestCase):

    def test_with_cycle(self):
        graph = DiGraph(vertices=("a", "b", "c", "d", "e", "f", "g"),
                        edges=(("a", "e"), ("c", "f"), ("c", "e"), ("c", "d"),
                               ("d", "f"), ("e", "d"), ("e", "b"), ("f", "a"),
                               ("b", "g"), ("b", "a")))

        with self.assertRaises(ValueError):
            graph.topological_sort()

    def test_without_cycle(self):
        graph = DiGraph(vertices=("a", "b", "c", "d", "e", "f"),
                        edges=(("a", "d"), ("f", "b"), ("b", "d"), ("f", "a"), ("d", "c")))

        expected = ("f", "e", "b", "a", "d", "c")
        self.assertEqual(expected, tuple(graph.topological_sort()))


# 4.8
def first_ancestor(root, a, b):
    if not root:
        return

    left_a = dfs(root.left, a)
    right_a = False
    if not left_a:
        right_a = dfs(root.right, a)

    left_b = dfs(root.left, b)
    right_b = False
    if not left_b:
        right_b = dfs(root.right, b)

    if left_a and left_b:
        return first_ancestor(root.left, a, b)
    elif not left_a and not left_b:
        return first_ancestor(root.right, a, b)

    if not (left_a or right_a) or not (left_b or right_b):
        return None

    return root


def dfs(node, a, b):
    if not node:
        return

    if node.val == a.val or node.val == b.val:
        return node

    left = dfs(node.left, a, b)
    right = dfs(node.right, a, b)
    if left and right:
        return node

    return left or right


class TestFirstAncestor(unittest.TestCase):

    def test_small(self):
        root = TreeNode(8, left=TreeNode(17), right=TreeNode(24))

        a = TreeNode(17)
        b = TreeNode(24)
        self.assertEqual(root, dfs(root, a, b))

    def test_bigger(self):
        root = TreeNode(5, left=TreeNode(8, left=TreeNode(17),
                                            right=TreeNode(24,
                                                right=TreeNode(94,
                                                left=TreeNode(108)))),
                           right=TreeNode(12, right=TreeNode(36,
                                                left=TreeNode(47),
                                                right=TreeNode(82))))

        a = TreeNode(94)
        b = TreeNode(17)
        self.assertEqual(root.left, dfs(root, a, b))

        a = TreeNode(47)
        b = TreeNode(17)
        self.assertEqual(root, dfs(root, a, b))

        a = TreeNode(47)
        b = TreeNode(82)
        self.assertEqual(root.right.right, dfs(root, a, b))

        a = TreeNode(47)
        b = TreeNode(36)
        self.assertEqual(root.right.right, dfs(root, a, b))


    @unittest.expectedFailure
    def test_non_existent(self):
        """This method doesn't support inexistent nodes."""
        a = TreeNode(900)
        b = TreeNode(17)
        # 900 is not in the tree.
        self.assertEqual(None, dfs(root, a, b))


# 4.9
def all_sequences(node):
    if not node:
        return [[]]

    left, right = all_sequences(node.left), all_sequences(node.right)

    weaved = []
    for left_val in left:
        for right_val in right:
            weaved = weave(left_val, right_val, [node.val], weaved)
    return weaved


def weave(left, right, prefix, results):
    print(f"left: {left}, right: {right}")
    if not (left and right):
        print("I am in here")
        result = prefix.copy()
        result.extend(left)
        result.extend(right)
        results.append(result)
        return results

    print(f"current left: {left}, current right: {right}")

    head = left[0]
    prefix.append(head)
    results = weave(left[1:], right, prefix, results)
    prefix.pop()
    head = right[0]
    prefix.append(head)
    results = weave(left, right[1:], prefix, results)
    prefix.pop()
    return results

class TestSequences:

    def test_small(self):
        root = TreeNode(20, left=TreeNode(9, left=TreeNode(5), right=TreeNode(12)),
                            right=TreeNode(25))

        seqs = all_sequences(root)
        for sequence in seqs:
            print(sequence)

    def test_large(self):
        root = TreeNode(25,
                        left=TreeNode(10,
                            left=TreeNode(5,
                                left=TreeNode(1)),
                            right=TreeNode(12)),
                        right=TreeNode(45,
                            left=TreeNode(32,
                                right=TreeNode(37))))

        seqs = all_sequences(root)
        for sequence in seqs:
            print(sequence)

# 4.10
def subtree_checker(root, node):
    if not root:
        return False
    elif root.val == node.val:
        return check_equals(root, node)

    return subtree_checker(root.left, node) or subtree_checker(root.right, node)

def check_equals(root, node):
    t1_stack, t2_stack = [root], [node]

    while t1_stack and t2_stack:
        t1_node, t2_node = t1_stack.pop(), t2_stack.pop()

        if not t1_node.val == t2_node.val:
            return False

        if t1_node.right:
            t1_stack.append(t1_node.right)
        if t1_node.left:
            t1_stack.append(t1_node.left)
        if t2_node.right:
            t2_stack.append(t2_node.right)
        if t2_node.left:
            t2_stack.append(t2_node.left)

    return not (bool(t1_stack) or bool(t2_stack))


class TestSubtreeChecker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.t1 = TreeNode(17,
                left=TreeNode(8,
                    left=TreeNode(9,
                        left=TreeNode(11),
                        right=TreeNode(23)
                        ),
                    right=TreeNode(12,
                        left=TreeNode(18),
                        right=TreeNode(40)
                        )
                    ),
                right=TreeNode(4,
                    left=TreeNode(16,
                        left=TreeNode(32),
                        right=TreeNode(7)
                        ),
                    right=TreeNode(42,
                        left=TreeNode(14),
                        right=TreeNode(27)
                        )
                    )
                )

    def test_existing(self):
        t2 = TreeNode(42,
                left=TreeNode(14),
                right=TreeNode(27)
                )

        self.assertTrue(subtree_checker(self.t1, t2))

    def test_not_existing(self):
        t2 = TreeNode(42,
                left=TreeNode(8),
                right=TreeNode(27)
                )

        self.assertFalse(subtree_checker(self.t1, t2))

    def test_extra_nodes_t2(self):
        t2 = TreeNode(9,
                left=TreeNode(11),
                right=TreeNode(23,
                    left=TreeNode(42))
                )

        self.assertFalse(subtree_checker(self.t1, t2))


# 4.11
class BinaryTree:
    """Insert and delete methods would require O(n) time as keys list needs to be updated.
       Find method would also be O(n) as the tree is not necessarily a binary search tree.

       This is a method that would work well for binary trees but a more efficient method
       as detailed in the book works for binary search trees where we can get to the smallest
       key very quickly. In essence, we could number the nodes based on their order in the tree
       and do a random choice for the number with a 1/N probability for each number. Following
       that, a simple traversal is required to find the node itself.
    """

    def __init__(self):
        self.keys = []

    def getRandomNode(self):
        return random.choice(self.keys)

# 4.12
def sum_paths(root, value):
    stack = [root]
    paths = 0
    while stack:
        node = stack.pop()
        paths += _dfs(node, value)

        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return paths

def _dfs(node, value, current_sum=0):
    if not node:
        return 0

    current_sum += node.val
    total_paths = 1 if current_sum == value else 0
    total_paths += _dfs(node.left, value, current_sum)
    total_paths += _dfs(node.right, value, current_sum)

    return total_paths

class TestPathFinder(unittest.TestCase):

    def test_small(self):
        root = TreeNode(4,
                left=TreeNode(-1,
                    left=TreeNode(2)
                    ),
                right=TreeNode(-4,
                    left=TreeNode(5)
                    )
                )

        self.assertEqual(sum_paths(root, 5), 3)
        self.assertEqual(sum_paths(root, 0), 1)
        self.assertEqual(sum_paths(root, 19), 0)



if __name__ == "__main__":
    unittest.main()
