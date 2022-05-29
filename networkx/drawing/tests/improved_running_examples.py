from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from networkx.drawing import hypergraph_layout
from networkx.drawing import our_layout


def random_hypergraph(num_of_vtx, num_of_edges):
    import random
    vtx = list(range(num_of_vtx))
    edges = []
    for edge in range(num_of_edges):
        v = set()
        num_of_vtx_in_edge = random.randint(1, num_of_vtx)
        for _ in range(num_of_vtx_in_edge):
            rand_vtx = random.randint(0, num_of_vtx - 1)
            v.add(rand_vtx)
        E = hypergraph_layout.hyperedge(list(v))
        edges.append(E)
    return hypergraph_layout.hypergraph(vtx, edges)


def run(num_of_vtx, num_of_edges):
    import time
    G = random_hypergraph(num_of_vtx, num_of_edges)
    start1 = time.perf_counter()
    our_layout.force_directed_hyper_graphs_using_social_and_gravity_scaling(G, seed=1, with_threads=True)
    finish1 = time.perf_counter()
    start2 = time.perf_counter()
    our_layout.force_directed_hyper_graphs_using_social_and_gravity_scaling(G, seed=1, with_threads=False)
    finish2 = time.perf_counter()
    print("------------------")
    print(f"Graph with {num_of_vtx} vertices, {num_of_edges} edges")
    print(f"without threads took {(finish1 - start1)} seconds")
    print(f"with threads took {(finish2 - start2)} seconds")
    return (finish1 - start1), (finish2 - start2)


if __name__ == '__main__':
    # on each test number of vertices is twice as many as the number of edges
    # what determinate the size of the input is the number of edges
    # I took 100 samples each time increased the number of edges by 5
    num_of_samples = 30
    with_scatters = []
    without_scatters = []
    for i in range(1, num_of_samples+1):
        without_threads, with_threads = run(num_of_vtx=i * 10, num_of_edges=i * 5)
        with_scatters.append(with_threads)
        without_scatters.append(without_threads)
    fig, ax = plt.subplots()
    plt.xlabel("Number of edges")
    plt.ylabel("Time in seconds")
    plt.title("Graph for comparison with and without thread improvement")
    for i, j in enumerate(range(1, num_of_samples)):
        plt.plot([j * 5, (j + 1) * 5], [with_scatters[i], with_scatters[i + 1]], 'r')
        plt.plot([j * 5, (j + 1) * 5], [without_scatters[i], without_scatters[i + 1]], 'b')
    red_patch = mpatches.Patch(color='red', label='with thread improvement')
    blue_patch = mpatches.Patch(color='blue', label='without thread improvement')
    ax.legend(handles=[red_patch, blue_patch])
    plt.show()
