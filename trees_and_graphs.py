import unittest
from collections import deque
from dataclasses import dataclass, field
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



if __name__ == "__main__":
    unittest.main()
