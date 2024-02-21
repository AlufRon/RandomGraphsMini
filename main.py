from collections import deque
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.backends.backend_pdf import PdfPages

def generate_random_graph(n, p):
    mean = n * (n - 1) // 2 * p
    variance = n * (n - 1) // 2 * p * (1 - p)
    M = np.random.normal(mean, np.sqrt(variance))
    M = int(round(M))  # Ensure M is an integer
    G = nx.gnm_random_graph(n, M)
    return G

def find_connected_components(graph):
    component_sizes = [len(c) for c in nx.connected_components(graph)]
    return component_sizes

def theoretical_case_1(n):
    return 2.5 * np.log(n)

def theoretical_case_2(n, lambda_):
    return n ** (2 / 3)

def theoretical_case_3(n, lambda_):
    return n ** (2 / 3)

def theoretical_case_4_L1(n):
    return n ** (1 / 3)

def theoretical_case_4_L2(n):
    return n ** (1 / 3)

def theoretical_case_5(n, c):
    from scipy.optimize import fsolve
    y = fsolve(lambda y: np.exp(-c * y) - 1 / y, 1)[0]
    return y * n

def plot_component_sizes(n_values, k_largest_component_sizes, theoretical_fn, d, lambda_=None, case_name=None, pdf_pages=None):
    plt.figure(figsize=(10, 6))
    for i, component_sizes in enumerate(k_largest_component_sizes, start=1):
        plt.plot(n_values, component_sizes, 'o-', label=f'{i} Largest Component')
    if lambda_ is not None:
        plt.plot(n_values, [d * theoretical_fn(n, lambda_) for n in n_values], 'r-', label='Theoretical')
    else:
        plt.plot(n_values, [d * theoretical_fn(n) for n in n_values], 'r-', label='Theoretical')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Component Size')
    plt.legend()
    plt.grid(True)
    plt.title(f'Number of Nodes vs. Component Size ({case_name})')
    if pdf_pages is not None:
        pdf_pages.savefig()
    else:
        plt.show()

def run_case(num_trials, c, k, p_fn, theoretical_fn, lambda_=None, isLambdaN=None, case_name=None, pdf_pages=None):
    n_values = range(10000, 100000, 10000)  # Adjust as needed
    k_largest_component_sizes = [[] for _ in range(k)]  # Initialize list of lists

    for n in n_values:
        component_sizes_trials = [[] for _ in range(k)]  # Initialize list of lists for each trial

        for _ in range(num_trials):
            if isLambdaN:
                lambda_ = n ** 0.01
            p = p_fn(n, c, lambda_)  # Calculate p based on the provided function
            graph = generate_random_graph(n, p)
            component_sizes = sorted(find_connected_components(graph), reverse=True)  # Sort in descending order
            for i in range(k):
                if i < len(component_sizes):  # Check if there is an i-th component
                    component_sizes_trials[i].append(component_sizes[i])  # Get the size of the i-th largest component
                else:
                    component_sizes_trials[i].append(0)  # If there is no i-th component, append 0

        for i in range(k):
            k_largest_component_sizes[i].append(np.mean(component_sizes_trials[i]))  # Average over trials

    plot_component_sizes(n_values, k_largest_component_sizes, theoretical_fn, 1, lambda_, case_name, pdf_pages)

def main():
    num_trials = 5  # Define the number of trials for each case
    k = 4  # Define the number of largest components to consider

    # Define the probability calculation functions for each case
    p_fn_case1 = lambda n, c, lambda_: c / n
    p_fn_case2 = lambda n, c, lambda_: (1 / n) - (lambda_ / n ** (4 / 3))
    p_fn_case3 = lambda n, c, lambda_: (1 / n) + (lambda_ / n ** (4 / 3))
    p_fn_case4 = lambda n, c, lambda_: 1 / n
    #p_fn_case5 = lambda n, c, lambda_: (1 + c) / n

    with PdfPages('output.pdf') as pdf_pages:
        # Run each case
        run_case(num_trials, 0.5, k, p_fn_case1, theoretical_case_1, case_name='Case 1 -', pdf_pages=pdf_pages)
        run_case(num_trials, 1.5, k, p_fn_case1, theoretical_case_1, case_name='Case 1 +', pdf_pages=pdf_pages)
        run_case(num_trials, 0, k, p_fn_case2, theoretical_case_2, 0.01, case_name='Case 2 - lambda', pdf_pages=pdf_pages)
        run_case(num_trials, 0, k, p_fn_case3, theoretical_case_3, 0.01, case_name='Case 3 + lambda', pdf_pages=pdf_pages)
        run_case(num_trials, 0, k, p_fn_case2, theoretical_case_2, 0, True, case_name='Case 2 - lambda n', pdf_pages=pdf_pages)
        run_case(num_trials, 0, k, p_fn_case3, theoretical_case_3, 0, True, case_name='Case 3 + lambda n', pdf_pages=pdf_pages)
        run_case(num_trials, 0.01, k, p_fn_case4, theoretical_case_4_L1, case_name='Case 4', pdf_pages=pdf_pages)
        #run_case(num_trials, 1.5, k, p_fn_case5, theoretical_case_5, case_name='Case 5', pdf_pages=pdf_pages)

if __name__ == "__main__":
    main()
