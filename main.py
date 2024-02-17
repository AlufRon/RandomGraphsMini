import networkx as nx
import numpy as np
import random
import random
import itertools
from networkx.generators.random_graphs import erdos_renyi_graph
import matplotlib


def generate_random_graph_binomial(n, p):
    mean = n * (n - 1) // 2 * p
    variance = n * (n - 1) // 2 * p * (1 - p)
    M = np.random.normal(mean, np.sqrt(variance))
    M = int(round(M))  # Ensure M is an integer
    G = nx.gnm_random_graph(n, M)
    return G


def getEpsilon(n):
    return np.power(n, (-1 / 3))


def connected_components_info(G):
    connected_components = nx.connected_components(G)

    component_sizes = []

    for component in connected_components:
        component_sizes.append(len(component))

    return len(component_sizes), component_sizes


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import fsolve
from collections import Counter


# Graph class
class Graph:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.graph = self.generate_random_graph()

    def generate_random_graph(self):
        return generate_random_graph_binomial(self.n, self.p)


# BFS function
def BFS(graph, start):
    visited = set()
    queue = [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(set(graph[vertex]) - visited)
    return visited


# Helper functions
def get_connectivity_components(graph):
    nodes = set(graph.nodes)
    connectivity_components = []
    while nodes:
        start_node = nodes.pop()
        component = BFS(graph, start_node)
        connectivity_components.append(component)
        nodes -= component
    return connectivity_components


def plot_connectivity_components(connectivity_components, n, case):
    sizes = sorted([len(component) for component in connectivity_components])
    plt.figure()
    plt.plot(sizes, '-o')
    plt.title(f'Case {case}: Sizes of connectivity components')
    plt.xlabel('Index of connectivity component')
    plt.ylabel('Size of connectivity component')
    plt.savefig(f'{case}_{n}.png')
    plt.close()


def compute_conjecture(case, n):
    if case == 'Very Subcritical':
        return np.log(n)
    elif case == 'Barely Subcritical':
        return np.log(n)
    elif case == 'Critical Window':
        return n ** (2 / 3)
    elif case == 'Barely Supercritical':
        return 2 * 0.01 * n * n ** (2 / 3)
    elif case == 'Very Supercritical':
        return fsolve(lambda y: np.exp(-2 * y) - 1 / y, 0.5) * n
    else:
        return None


def plot_connectivity_components_by_n(n_values, p_func):
    num_components = []  # Store the number of components for each n
    largest_component_sizes = []  # Store the size of the largest component for each n
    top_5_component_sizes = [[] for _ in range(5)]  # Store the sizes of the top 5 components for each n

    for n in n_values:
        p = p_func(n)
        G = generate_random_graph_binomial(n, p)
        num_components_current, component_sizes_current = connected_components_info(G)
        largest_component_size = max(component_sizes_current)

        num_components.append(num_components_current)
        largest_component_sizes.append(largest_component_size)

        # Sort the component sizes and get the top 5
        component_sizes_current.sort(reverse=True)
        for i in range(5):
            if i < len(component_sizes_current):
                top_5_component_sizes[i].append(component_sizes_current[i])
            else:
                top_5_component_sizes[i].append(0)  # If there are less than 5 components, fill with 0

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, num_components, label="Number of Components")
    plt.plot(n_values, largest_component_sizes, label="Size of Largest Component")

    # Plot the sizes of the top 5 components
    for i in range(5):
        plt.plot(n_values, top_5_component_sizes[i], label=f"Size of {i + 1}th Largest Component")

    plt.xlabel("Number of Nodes (n)")
    plt.ylabel("Number of Components/Size")
    plt.title(f"Connected Components in Random Graphs")
    plt.legend()
    plt.grid(True)
    plt.show()


# Define your p function
def p_func(n):
    epsilon = getEpsilon(n)
    return (1 / n) + epsilon


# Call the function with your desired range of n values
plot_connectivity_components_by_n(range(100000, 1000001, 100000), p_func)


# def generate_random_graph_random_sample(n, p):
#     all_edges = list(itertools.combinations(range(n), 2))
#
#     M = int(p * n * (n - 1) / 2)
#
#     edges = random.sample(all_edges, M)
#
#     graph = [[0] * n for _ in range(n)]
#
#     for i, j in edges:
#         graph[i][j] = graph[j][i] = 1
#
#     return graph
#


#
# def generate_random_graph(n, M):
#     graph = [[0] * n for _ in range(n)]
#
#     for _ in range(M):
#         while True:
#             i, j = random.randint(0, n - 1), random.randint(0, n - 1)
#             # Ensure the edge is not a self-loop and does not already exist
#             if i != j and graph[i][j] == 0:
#                 break
#         graph[i][j] = graph[j][i] = 1
#
#     return graph
#
#
# import random
#
#
def generate_random_graph(n, M):
    graph = [[0] * n for _ in range(n)]

    for _ in range(M):
        while True:
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            # Ensure the edge is not a self-loop and does not already exist
            if i != j and graph[i][j] == 0:
                break
        graph[i][j] = graph[j][i] = 1

    return graph
