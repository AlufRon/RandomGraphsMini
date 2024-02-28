import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def generate_random_graph(n, p):
    mean = n * (n - 1) // 2 * p
    variance = n * (n - 1) // 2 * p * (1 - p)
    M = np.random.normal(mean, np.sqrt(variance))
    M = int(round(M))
    G = nx.gnm_random_graph(n, M)
    return G


def find_connected_components(graph):
    component_sizes = [len(c) for c in nx.connected_components(graph)]
    return component_sizes


def theoretical_case_1(n):
    return 2.3 * np.log(n)


def theoretical_case_2(n, lambda_):
    n_lambda_ = n ** lambda_
    return 2 * (n ** (2 / 3) * ((n_lambda_ ** -2) * np.log(n_lambda_)))


def theoretical_case_3(n):
    return 0.5 * (n ** (2 / 3))


def theoretical_case_4_L1(n, lambda_):
    n_lambda_ = n ** lambda_
    return (2 * n_lambda_) * (n ** (2 / 3))


def theoretical_case_4_L2(n, lambda_):
    n_lambda_ = n ** lambda_
    return ((n ** 0.01) ** -2) * (n ** (2/3)) * np.log(n ** 0.01)



def theoretical_case_5_L1(n):
    from scipy.optimize import fsolve
    y = fsolve(lambda y: np.exp(-2 * y) - 1 / y, 1)[0]  # c = 2 in np.exp(-2 * y)
    return y * n


def theoretical_case_5_L2(n):
    return np.log(n)



def plot_component_sizes(n_values, k_largest_component_sizes, theoretical_fn, d, lambda_=None, case_name=None,
                         pdf_pages=None, k_th_component=None):
    plt.figure(figsize=(10, 6))
    for i, component_sizes in enumerate(k_largest_component_sizes, start=1):
        if k_th_component is not None:
            plt.plot(n_values, component_sizes, 'o-', label=f'{k_th_component} Largest Component')
        else:
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

 
def run_case(num_trials, c, k, p_fn, theoretical_fn, lambda_=None, isLambdaN=None, case_name=None, pdf_pages=None, k_th_component=None):
    n_values = range(100000, 1000000, 100000)
    k_largest_component_sizes = [[] for _ in range(k)]
    for n in n_values:
        component_sizes_trials = [[] for _ in range(k)]

        for _ in range(num_trials):
            if isLambdaN:
                lambda_ = n ** 0.01
            p = p_fn(n, c, lambda_)
            graph = generate_random_graph(n, p)
            component_sizes = sorted(find_connected_components(graph), reverse=True)
            for i in range(k):
                if i < len(component_sizes):
                    component_sizes_trials[i].append(component_sizes[i])
                else:
                    component_sizes_trials[i].append(0)

        for i in range(k):
            k_largest_component_sizes[i].append(np.mean(component_sizes_trials[i]))

    if k_th_component:
        plot_component_sizes(n_values, [k_largest_component_sizes[k_th_component - 1]], theoretical_fn, 1, lambda_, case_name, pdf_pages, k_th_component)
        return
    plot_component_sizes(n_values, k_largest_component_sizes, theoretical_fn, 1, lambda_, case_name, pdf_pages)

def main():
    num_trials = 5
    k = 5

    p_fn_case1 = lambda n, c, lambda_: c / n
    p_fn_case2 = lambda n, c, lambda_: (1 / n) - (lambda_ / n ** (4 / 3))
    p_fn_case3_sub = lambda n, c, lambda_: (1 / n) - (2 / n ** (4 / 3))
    p_fn_case3_add = lambda n, c, lambda_: (1 / n) + (2 / n ** (4 / 3))
    p_fn_case4 = lambda n, c, lambda_: (1 / n) + (lambda_ / n ** (4 / 3))
    p_fn_case5 = lambda n, c, lambda_: c / n

    with PdfPages('output.pdf') as pdf_pages:
        # case 1 c = 1
        run_case(num_trials, 0.5, 1, p_fn_case1, theoretical_case_1,
                 case_name='Case 1 - assumption 1 : Very Subcritical', pdf_pages=pdf_pages)
        run_case(num_trials, 0.5, 8, p_fn_case1, theoretical_case_1,
                 case_name='Case 1 - assumption 2 : Very Subcritical', pdf_pages=pdf_pages)
        # case 2
        run_case(num_trials, 0, 1, p_fn_case2, theoretical_case_2, 0.01,
                 case_name='Case 2 - assumption 1 : Barely Subcritical', pdf_pages=pdf_pages)
        run_case(num_trials, 0, 6, p_fn_case2, theoretical_case_2, 0.01,
                 case_name='Case 2 - assumption 2 : Barely Subcritical', pdf_pages=pdf_pages)

        # case 3
        run_case(num_trials, 0, 1, p_fn_case3_sub, theoretical_case_3, case_name='Case 3 The Critical Window: Subtract',
                 pdf_pages=pdf_pages)
        run_case(num_trials, 0, 1, p_fn_case3_add, theoretical_case_3, case_name='Case 3 The Critical Window Addition',
                 pdf_pages=pdf_pages)

        # case 4
        run_case(num_trials, 0, 1, p_fn_case4, theoretical_case_4_L1, 0.01, True,
                 case_name='Case 4 - assumption 1: Barely Supercritical',
                 pdf_pages=pdf_pages)
        run_case(num_trials, 0, 2, p_fn_case4, theoretical_case_4_L2, 0.01, True,
                 case_name='Case 4 - assumption 2: Barely Supercritical',
                 pdf_pages=pdf_pages)

        # case 5 -  c = 2
        run_case(num_trials, 2, 1, p_fn_case5, theoretical_case_5_L1,
                 case_name='Case 5 - assumption 1 : Very Supercritical', pdf_pages=pdf_pages)
        run_case(num_trials, 2, 2, p_fn_case5, theoretical_case_5_L2,
                 case_name='Case 5 - assumption 2 : Very Supercritical', pdf_pages=pdf_pages, k_th_component=2)


if __name__ == "__main__":
    main()
