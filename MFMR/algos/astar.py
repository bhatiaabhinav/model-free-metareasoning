from utils import Node, OpenList, get_children_nodes, get_key, get_solution

def solve(problem):
    start_node = Node(problem.start_state)
    start_node_value = start_node.path_cost + problem.get_heuristic(start_node.state)

    open_list = OpenList()
    open_list.add(start_node, start_node_value)

    closed_list = set()

    while open_list:
        current_node = open_list.remove()

        if problem.is_goal(current_node.state):
            return get_solution(current_node)

        current_node_key = get_key(current_node.state)
        closed_list.add(current_node_key)

        for child_node in get_children_nodes(problem, current_node):
            child_node_key = get_key(child_node.state)

            if child_node_key not in closed_list and child_node not in open_list:
                child_node_value = child_node.path_cost + problem.get_heuristic(child_node.state)
                open_list.add(child_node, child_node_value)
            elif child_node in open_list:
                stored_child_node = open_list[child_node]

                child_node_value = child_node.path_cost + problem.get_heuristic(child_node.state)
                stored_child_node_value = stored_child_node.path_cost + problem.get_heuristic(stored_child_node.state)

                if child_node_value < stored_child_node_value:
                    del open_list[stored_child_node]
                    open_list.add(child_node, child_node_value)

    return None
