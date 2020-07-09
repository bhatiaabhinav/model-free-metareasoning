from utils import Node, OpenList, get_children_nodes, get_key, get_solution


def solve(problem, weight):
    start_node = Node(problem.start_state)
    start_node_key = get_key(start_node.state)
    start_node_value = start_node.path_cost + \
        weight * problem.get_heuristic(start_node.state)

    open_list = OpenList()
    open_list.add(start_node, start_node_value)

    closed_set = set()
    best_value = float('inf')

    generated_states = set()
    generated_states.add(start_node_key)

    path_costs = {
        start_node_key: start_node.path_cost
    }

    while open_list:
        current_node = open_list.remove()
        current_node_value = current_node.path_cost + \
            problem.get_heuristic(current_node.state)

        if current_node_value < best_value:
            current_node_key = get_key(current_node.state)
            closed_set.add(current_node_key)

            for child_node in get_children_nodes(problem, current_node):
                child_node_value = child_node.path_cost + \
                    problem.get_heuristic(child_node.state)
                child_node_key = get_key(child_node.state)

                if child_node_value < best_value:
                    if problem.is_goal(child_node.state):
                        path_costs[child_node_key] = child_node.path_cost
                        best_value = child_node_value

                        solution = get_solution(child_node)
                        cost = len(solution)

                        print({'cost': cost, 'solution': solution})
                    elif child_node_key in closed_set or child_node in open_list:
                        if path_costs[child_node_key] > child_node.path_cost:
                            path_costs[child_node_key] = child_node.path_cost
                            value = path_costs[child_node_key] + weight * \
                                problem.get_heuristic(child_node.state)

                            if child_node_key in closed_set:
                                closed_set.remove(child_node_key)

                            open_list.add(child_node, value)
                    else:
                        path_costs[child_node_key] = child_node.path_cost
                        value = path_costs[child_node_key] + weight * \
                            problem.get_heuristic(child_node.state)
                        open_list.add(child_node, value)

    return None
