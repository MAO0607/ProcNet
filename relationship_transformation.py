import re


def process_workflow(fragments):
    fragments = fragments.splitlines()

    edges = []
    current_fragment = []
    for line in fragments:
        line = line.strip()

        if re.match(r'fragment\d+:', line):
            if current_fragment:
                edges.append(current_fragment)
                current_fragment = []
            continue

        matches = re.findall(r'\[(.*?)\]', line)
        for match in matches:
            items = [item.strip().replace(r"\'", "'") for item in match.split(',')]
            current_fragment.append(items)


    if current_fragment:
        edges.append(current_fragment)

    result = []
    for z_edges in edges:
        from collections import defaultdict
        graph = defaultdict(list)
        nodes = set()
        for a, b in z_edges:
            graph[a].append(b)
            nodes.update({a, b})

        def find_paths():
            in_degree = {n: 0 for n in nodes}
            for _, b in z_edges:
                in_degree[b] += 1
            starts = [n for n, cnt in in_degree.items() if cnt == 0]

            all_paths = []

            def dfs(node, path):
                path.append(node)
                if not graph[node]:
                    all_paths.append(list(path))
                for neighbor in graph[node]:
                    dfs(neighbor, path)
                path.pop()

            for start in starts:
                dfs(start, [])
            return all_paths

        all_paths = find_paths()

        branch_dict = {}
        for i, path in enumerate(all_paths, 1):
            intermediates = path[1:-1]
            branch_dict[f'branch{i}'] = intermediates


        branches = list(branch_dict.values())
        end_nodes = {path[-1] for path in all_paths}

        for i in range(len(branches)):
            for j in range(len(branches)):
                if i != j:
                    for src in branches[i]:
                        for dst in branches[j]:
                            if dst not in end_nodes:
                                result.append([src, dst])

    return result
