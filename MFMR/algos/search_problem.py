import queue

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


class PriorityDict:
    '''A hybrid data structure which efficiently combines functionalities of a priority queue and a dictionary'''

    def __init__(self, tie_breaker='FIFO'):
        # used to mark a key-value pair as deleted in O(1) time.
        self.DELETED_VALUE = object()
        self._ke_map = {}  # maps legit keys to entry
        self._ek_map = {}  # maps entry id to key
        # the priority queue to hold entries in (priority, id, value) format
        self._pq = queue.PriorityQueue()
        '''assign a unique id to every value to break ties'''
        self._counter = 0
        self._increment = 1 if tie_breaker == 'FIFO' else -1

    def add(self, key, value, priority):
        '''Adds a new key-value pair with given priority'''
        assert key not in self._ke_map, "Key already present"
        entry = [priority, self._counter, value]
        self._counter += self._increment
        self._pq.put(entry)
        self._ke_map[key] = entry
        self._ek_map[entry[1]] = key

    def pop(self):
        '''retrieve lowest priority entry and remove it. Returns (key, value, priority) tuple'''
        while not self._pq.empty():
            entry = self._pq.get()
            if entry[-1] == self.DELETED_VALUE:
                # automatically popped. No housekeeping needed in other data structs.
                assert entry[1] not in self._ek_map, "A deleted entry found in hashmap"
                continue
            else:
                key = self._ek_map[entry[1]]
                del self._ke_map[key]
                del self._ek_map[entry[1]]
                return key, entry[-1], entry[0]
        raise KeyError('The queue is empty')

    def get(self, key, default=None):
        '''retrieve an entry (value, priority) based on key. Returns `default` if key not found'''
        if key in self._ke_map:
            entry = self._ke_map[key]
            assert entry[-1] != self.DELETED_VALUE, "How did this happen. Item deleted but present in hashmap"
            return entry[-1], entry[0]
        else:
            return default

    def __getitem__(self, key):
        '''retrieve an entry (value, priority) based on key. Raises KeyError if key not present'''
        entry = self._ke_map[key]
        assert entry[-1] != self.DELETED_VALUE, "How did this happen. Item deleted but present in hashmap"
        return entry[-1], entry[0]

    def delete(self, key):
        '''deletes an entry based on key. Does nothing if key not found.'''
        entry = self._ke_map.get(key)
        if entry is not None:
            assert entry[-1] != self.DELETED_VALUE, "How did this happen. Item deleted but present in hashmap"
            del self._ke_map[key]
            del self._ek_map[entry[1]]
            entry[-1] = self.DELETED_VALUE

    def __delitem__(self, key):
        '''deletes an entry based on key. Raises Keyerror if key not present'''
        entry = self._ke_map[key]
        if entry is not None:
            assert entry[-1] != self.DELETED_VALUE, "How did this happen. Item deleted but present in hashmap"
            del self._ke_map[key]
            del self._ek_map[entry[1]]
            entry[-1] = self.DELETED_VALUE

    def __contains__(self, key):
        '''checks if there is an entry with given key'''
        return key in self._ke_map

    def __len__(self):
        assert len(self._ke_map) == len(
            self._ek_map), "Something is wrong. Length of ke and ek maps are different"
        return len(self._ke_map)
