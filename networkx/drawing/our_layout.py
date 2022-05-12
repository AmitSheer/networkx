import math
import random

import networkx as nx
import matplotlib.pyplot as plt

from networkx.utils import np_random_state

__all__ = [
    "force_directed_hyper_graphs_using_social_and_gravity_scaling",
]


def rep(x, k):
    return k ** 2 / x ** 2


def att(x, k):
    return x / k


def fd(G: nx.Graph, seed: int, iterations: int = 50, threshold=70e-4):
    import numpy as np
    A = nx.to_numpy_array(G)
    k = math.sqrt(1 / len(A))
    # k = 1
    np.random.seed(seed)
    pos = np.asarray(np.random.rand(len(A), 2))
    # I = np.zeros(shape=(len(A), 2), dtype=float)
    I = np.zeros(shape=(2, len(A)), dtype=float)
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    old_t = t
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    gamma_t = 0
    mass = [v for v in nx.closeness_centrality(G).values()]
    center = (np.sum(pos, axis=0) / len(pos))

    for iteration in range(iterations):
        I *= 0
        for v in range(len(A)):
            delta = (pos[v] - pos).T
            distance = np.sqrt((delta ** 2).sum(axis=0))
            distance = np.where(distance < 0.01, 0.01, distance)
            Ai = A[v]
            # displacement "force"
            I[:, v] += (
                               delta * (k * k / distance ** 2 - Ai * distance / k)
                       ).sum(axis=1) + gamma_t * mass[v] * (center - pos[v])
        length = np.sqrt((I ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (I * t / length).T
        pos += delta_pos
        # cool temperature
        t -= dt
        if gamma_t > 125:
            print('gamma')
            break
        if (np.linalg.norm(delta_pos) / len(A)) < threshold:
            print("adasds")
            print(iteration)
            threshold /= 3
            # break
            gamma_t += 6 * round(iteration / 200)
            # threshold *= 10
            # break
            # gamma_t += 0.2
            # pp = {}
            # for i in range(len(pos)):
            #     pp[np.array(g.nodes)[i]] = np.array(pos[i])
            # nx.draw(g, pp, node_size=70)
            # # print(np.zeros(shape=(5, 2), dtype=float))
            # plt.show()
            # t = old_t
        iteration += 1
        #     repulsion
        #     for u in range(len(A)):
        #
        #         delta = pos[v] - pos[u]
        #         # Repulsion Force
        #         if v != u:
        #             dist = np.linalg.norm(delta)
        #             I[v] += rep(dist, k) * delta
        #         # attraction
        #         if A[v][u] == 1:
        #             dist = np.linalg.norm(delta)
        #             I[v] -= att(dist, k) * delta
        # length = np.sqrt((I ** 2).sum(axis=0))
        # length = np.where(length < 0.01, 0.1, length)
        # delta_pos = I*t/length
        # pos += delta_pos
        # t -= dt
        # if (np.linalg.norm(delta_pos) / len(A)) < threshold:
        #     break

        #     pos[v] += 0.1 * np.clip(I[v], a_min=-10, a_max=10)
    print(iteration)
    return pos


# @nx.not_implemented_for("directed")
def force_directed_hyper_graphs_using_social_and_gravity_scaling(G, k=None, pos=None, iterations=50,
                                                                 threshold=1e-4, centrality_type=0, graph_type=0,
                                                                 seed=1):
    """Positions nodes using Fruchterman-Reingold force-directed algorithm combined with Hyper-Graphs and Social and
    Gravitational Forces.

    Receives a Hyper-Graph and changes it to be a normal graph. Then calculates the Value of each node according to
    Social centrality Parameters. Nodes with high value are counted as central nodes in the graph and are more attracted
    to the center of the graph dimension.

    After running the algorithm the pos will be updated to reflect the social and force-directed values of the nodes.


    Parameters
    ----------
    G : graph
      A NetworkX graph.

    k : float (default=None)
        Optimal distance between nodes.  If None the distance is set to
        1/sqrt(n) where n is the number of nodes.  Increase this value
        to move nodes farther apart.

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        random initial positions.

    iterations : int  optional (default=50)
        Maximum number of iterations taken

    threshold: float optional (default = 1e-4)
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.

    centrality_type: int optional (default=0)
        Centrality type for the Social gravity field used in the algorithm.

    graph_type: int optional (default=0)
        Graph type for chosing type of conversion from hyper-graph to graph (cycle/wheel/star/complete)


    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Notes
    -----
    This algorithm currently only works on hyper-graphs.

    The algorithm is based on the work of Fruchterman-Reingold and adding Forces that mimic the Social interaction of
    social networks. Forces such as closeness, betweenness and degree centrality. using these forces to force place the
    nodes of the graph in a circular way and minimize the space used by the graph in pointing it.


    References
    ----------
    .. [1] Michael J. Bannister, David Eppstein, Michael T. Goodrich and Lowell Trott:
       Force-Directed Graph Drawing Using Social Gravity and Scaling.
       Graph Drawing. GD 2012. Lecture Notes in Computer Science, vol 7704. Springer, Berlin, Heidelberg.
       https://doi.org/10.1007/978-3-642-36763-2_37
    .. [2] Naheed Anjum Arafat and St´ephane Bressan:
       Hypergraph Drawing by Force-Directed Placement
       DEXA 2017. Lecture Notes in Computer Science(), vol 10439. Springer, Cham.
      https://doi.org/10.1007/978-3-319-64471-4_31

    """

    # p is a list that represent the current position of a vertex
    # m is a list that represent the mass of a vertex
    # i is a list that represent the movement direction
    # graph is the initial graph (boolean 2D array that represent edges)
    from itertools import count
    import numpy as np

    A = nx.to_numpy_array(G)

    if k is None:
        k = np.sqrt(1.0 / len(A))
    # randomize positions
    if pos is None:
        pos = np.random.rand(len(A), 2)
        # np.round(pos, 2)

    else:
        pos = np.array(pos, dtype=np.dtype(float))

    # adjacent mapping
    # TODO: find a way to calculate the delta better
    delta = 1.15
    #     m = compute_mass_centrality(graph)
    m = np.array([v for v in nx.closeness_centrality(G).values()])
    sigma = 0.01
    i_max: float = 5.
    gamma_t = 0
    xi = np.sum(pos, axis=0) / len(pos)
    attraction_equation = lambda pos_u, pos_v: ((np.linalg.norm(pos_u - pos_v, axis=-1) / k) * (pos_u - pos_v))
    repulsion_equation = lambda pos_u, pos_v: ((k * k) / (np.linalg.norm(pos_u - pos_v) ** 2)) * (
            pos_v - pos_u)
    gravitation_equation = lambda pos_v, m_v: gamma_t * m_v * (xi - pos_v)
    for t in range(iterations):
        i = np.array([[0, 0]] * len(A), dtype=np.dtype(float))

        # TODO: check if the calculation of xi need to be done outside of the main loop or in it
        for v in range(len(pos)):
            repulsion, attraction, gravitation = 0, 0, 0
            for u in range(len(pos)):
                if v == u:
                    continue
                # dist = pos[u] - pos[v]
                # sub = pos[u] - pos[v]
                # repulsion = repulsion + (k ** 2 / (dist ** 2)) * sub
                repulsion = repulsion + repulsion_equation(pos[u], pos[v])
                if A[v][u] == 1.:  # there is an edge
                    attraction = attraction + attraction_equation(pos[u], pos[v])
                # gravitation = gravitation + gamma_t * m[v] * (xi - pos[v])
                gravitation = gravitation + gravitation_equation(pos[v], m[v])
            i[v] = attraction + repulsion + gravitation
        for v in range(0, len(pos)):
            pos[v] = pos[v] + sigma * np.array([(min(i_max, float(i[v][0]))), min(i_max, float(i[v][1]))])
            # pos[v] = pos[v] + sigma * np.array([(min(i_max, float(i[v][0]))), min(i_max, float(i[v][1]))])
            # print(pos[v])
        if np.max(i) - np.min(i) < delta:
            gamma_t = gamma_t + 0.2
            delta -= 0.1
        if gamma_t >= 2.5:
            break
    return pos


if __name__ == '__main__':
    import numpy as np

    # g = nx.Graph()
    # g.add_edge(0, 1)
    # g.add_edge(2, 3)
    # g.add_edge(4, 3)
    # g.add_edge(2, 1)
    # g.add_edge(0, 4)
    # g.add_edge(4, 5)
    # g.add_edge(6, 7)
    # # g.add_edge(5,6)
    # g.add_edge(7, 8)
    # g.add_edge(6, 8)
    # # for i in nx.closeness_centrality(g):
    # b = random.Random()
    # b.seed(1)
    # # g = nx.Graph()
    # while g.edges.__len__() != 40:
    #     d = b.randint(0, 30)
    #     a = b.randint(0, 30)
    #     if a != d and g.has_edge(a, d) is False:
    #         g.add_edge(a, d)
    # plt.show()
    # g.nodes.keys()
    g: nx.Graph = nx.random_tree(70,1)
    f: list[nx.Graph] = []
    for i in range(1, 5):
        f.append(nx.random_tree(70, i))
        # pos_nx = nx.spring_layout(f[i-1], iterations=700)
        # nx.draw(f[i-1], pos_nx, node_size=70)
        # # nx.draw_spring
        # plt.show()


    for i in range(1, 5):
        for j in f[i-1].edges:
            g.add_edge(j[0]+70*(i+3)+1, j[1]+70*(i+3)+1)

    pos_nx = nx.spring_layout(g, iterations=700)
    nx.draw(g, pos_nx, node_size=70)
    # nx.draw_spring
    plt.show()
    pos = fd(g, 1, iterations=1000)
    pos = nx.rescale_layout(pos)
    # pos = force_directed_hyper_graphs_using_social_and_gravity_scaling(g)
    pp = {}
    for i in range(len(pos)):
        pp[np.array(g.nodes)[i]] = np.array(pos[i])
    nx.draw(g, pp, node_size=50)
    # print(np.zeros(shape=(5, 2), dtype=float))
    plt.show()
