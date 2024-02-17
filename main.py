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
    return 5*np.power(n,(-1/3))


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
        return n**(2/3)
    elif case == 'Barely Supercritical':
        return 2 * 0.01 * n * n**(2/3)
    elif case == 'Very Supercritical':
        return fsolve(lambda y: np.exp(-2*y) - 1/y, 0.5) * n
    else:
        return None


def plot_connectivity_components_by_n(n_values, p):
    num_components = []  # Store the number of components for each n
    largest_component_sizes = []  # Store the size of the largest component for each n

    for n in n_values:
        p = 1 / n
        G = generate_random_graph_binomial(n, p)
        num_components_current, component_sizes_current = connected_components_info(G)
        largest_component_size = max(component_sizes_current)

        num_components.append(num_components_current)
        largest_component_sizes.append(largest_component_size)

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, num_components, label="Number of Components")
    plt.plot(n_values, largest_component_sizes, label="Size of Largest Component")
    plt.xlabel("Number of Nodes (n)")
    plt.ylabel("Number of Components/Size")
    plt.title(f"Connected Components in Random Graphs (p={p:.2f})")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    # pdf = FPDF()
    # cases = [
    #     {'c': 0.5, 'p_func': lambda n: 0.5 / n, 'name': 'Very Subcritical'},
    #     # Add other cases here
    # ]
    # for i, case in enumerate(cases, start=1):
    #     for n in range(1, 100001, 1000):  # Adjust the step size as needed
    #         p = case['p_func'](n)
    #         graph = Graph(n, p).graph
    #         connectivity_components = get_connectivity_components(graph)
    #         plot_connectivity_components(connectivity_components, n, case['name'])
    #         conjecture = compute_conjecture(case['name'], n)
    #         print(f'Computed conjecture for case {case["name"]}, n={n}: {conjecture}')
    #         pdf.add_page()
    #         pdf.set_font("Arial", size = 15)
    #         pdf.cell(200, 10, txt = f"Case: {case['name']}, n={n}", ln = True, align = 'C')
    #         pdf.cell(200, 10, txt = f"Computed conjecture: {conjecture}", ln = True, align = 'C')
    #         pdf.image(f'{case["name"]}_{n}.png', x=10, y=30, w=100)
    # pdf.output("report.pdf")
    # Example usage
    n_values = range(100, 100000, 10)  # 10 values of n from 10 to 90
    p = 0.5  # Set the edge probability

    plot_connectivity_components_by_n(n_values, p)
if __name__ == "__main__":
    main()



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




