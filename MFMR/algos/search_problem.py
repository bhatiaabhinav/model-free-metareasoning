import numpy as np


class SearchProblem:
    def __init__(self):
        self.start_state = None
        self.random = np.random.RandomState()

    def seed(self, seed):
        self.random.seed(seed)

    def reset(self):
        raise NotImplementedError()

    def get_children_nodes(self, parent):
        children_nodes = []
        for successor in self.successors(parent.state):
            path_cost = parent.path_cost + \
                self.cost(
                    parent.state, successor['action'], successor['state'])
            child_node = Node(
                successor['state'], parent, path_cost, parent.depth + 1, successor['action'])
            children_nodes.append(child_node)
        return children_nodes

    def goal_test(self, state):
        raise NotImplementedError()

    def successors(self, state):
        raise NotImplementedError()

    def cost(self, state, action, next_state):
        raise NotImplementedError()

    def heuristic(self, state):
        raise NotImplementedError()


def get_key(state):
    return str(state.tolist())


class Node:
    def __init__(self, state, parent=None, path_cost=0, depth=0, action=None):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.depth = depth
        self.action = action

    def get_solution(self):
        if self.parent is None:
            return []
        return self.parent.get_solution() + [self.action]


class OpenList(object):
    def __init__(self):
        self.items = []
        self.cache = {}

    def add(self, node, value):
        self.items.append((value, node))
        self.items.sort(key=lambda item: item[0])

        node_key = get_key(node.state)
        self.cache[node_key] = node

    def remove(self):
        return self.items.pop(0)[1]

    def __len__(self):
        return len(self.items)

    def __contains__(self, node):
        node_key = get_key(node.state)
        return node_key in self.cache

    def __getitem__(self, node):
        node_key = get_key(node.state)
        return self.cache[node_key]

    def __delitem__(self, new_node):
        new_node_key = get_key(new_node.state)

        for i, (_, node) in enumerate(self.items):
            node_key = get_key(node.state)

            if node_key == new_node_key:
                self.items.pop(i)

        del self.cache[new_node_key]
