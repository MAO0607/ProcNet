from collections import defaultdict, deque
from itertools import combinations

def parse_input(raw_edges):
    parsed = []
    for line in raw_edges:
        clean = line.strip().strip('[]').split(',', 1)
        src = clean[0].strip()
        dst = clean[1].strip()
        parsed.append([src, dst])
    return parsed


def find_precise_branches(edges):
    all_nodes = set()
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)
    for src, dst in edges:
        all_nodes.update([src, dst])
        graph[src].append(dst)
        reverse_graph[dst].append(src)

    branches = []
    visited_pairs = set()

    def get_reachable(start, max_depth=10):
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth > max_depth or node in visited:
                continue
            visited.add(node)
            for neighbor in graph[node]:
                queue.append((neighbor, depth + 1))
        return visited

    def find_convergence(a_path, b_path):
        a_nodes = {n for path in a_path for n in path}
        for node in reversed(b_path):
            if node in a_nodes:
                return node
        return None

    for node in list(graph.keys()):
        if len(graph[node]) < 2:
            continue

        for first, second in combinations(graph[node], 2):
            if (node, first, second) in visited_pairs:
                continue
            visited_pairs.update([(node, first, second), (node, second, first)])

            reach_first = get_reachable(first)
            reach_second = get_reachable(second)
            common_nodes = reach_first & reach_second

            if not common_nodes:
                continue

            def trace_path(start):
                path = []
                current = start
                while graph[current] and len(path) < 5:
                    next_node = graph[current][0]
                    path.append((current, next_node))
                    current = next_node
                return path

            path_a = trace_path(first)
            path_b = trace_path(second)

            convergence = None
            for i in range(min(len(path_a), len(path_b))):
                a_nodes = {edge[1] for edge in path_a[:i + 1]}
                b_nodes = {edge[1] for edge in path_b[:i + 1]}
                common = a_nodes & b_nodes
                if common:
                    convergence = common.pop()
                    break

            if convergence:
                branch_edges = []
                branch_edges.append((node, first))
                branch_edges.append((node, second))

                for edge in path_a:
                    branch_edges.append(edge)
                    if edge[1] == convergence:
                        break

                for edge in path_b:
                    branch_edges.append(edge)
                    if edge[1] == convergence:
                        break

                seen = set()
                unique_edges = []
                for edge in branch_edges:
                    if edge not in seen:
                        seen.add(edge)
                        unique_edges.append(list(edge))

                if len(unique_edges) > 2:
                    branches.append(unique_edges)

    return branches