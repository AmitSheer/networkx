'''First artical: Force-Directed Graph Drawing Using Social Gravity and Scaling
authots: Michael J. Bannister, David Eppstein, Michael T. Goodrich and Lowell Trott
link: https://arxiv.org/pdf/1209.0748.pdf

Second artical: Hypergraph Drawing by Force-Directed Placement
authots: Naheed Anjum Arafat and Stephane Bressan
link: https://www.researchgate.net/publication/318823299_Hypergraph_Drawing_by_Force-Directed_Placement

Our names: Amit Sheer Cohen and Neta Roth'''

import math

import numpy
import numpy as np
from scipy.spatial import ConvexHull

import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing import hypergraph_layout
from networkx.drawing.hypergraph_layout import hyperedge, hypergraph

import logging

# create and configure logger
logging.basicConfig(filename='my_logging.log', level=logging.INFO)
logger = logging.getLogger()


__all__ = [
    "force_directed_hyper_graphs_using_social_and_gravity_scaling",
    "force_directed"
]


def get_points_order(hull):
    order_by = hull.simplices[0]
    simplices = hull.simplices[1:]
    for i in range(hull.simplices.shape[0]):
        for j in range(0, simplices.shape[0]):
            if order_by[len(order_by) - 1] in simplices[j]:
                if simplices[j][0] == order_by[len(order_by) - 1]:
                    order_by = np.append(order_by, simplices[j][1])
                else:
                    order_by = np.append(order_by, simplices[j][0])
                simplices = np.delete(simplices, j, axis=0)
                break
    return order_by


def rep(x, k):
    return k ** 2 / x ** 2


def att(x, k):
    return x / k


def force_directed(G: nx.Graph, seed: int, iterations: int = 50, threshold=70e-4, centrality=None, gravity: int = 6):
    """

    Parameters
    ----------
    G: nx.Graph
     for easy calculations and usage of networkx functions
    seed: int
        Randomize the initial positions of nodes for consistent results
    iterations: int (default=50)
        maximum number of iterations the algorithm is allows to used
    threshold: float optional (default = 70e-4)
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold
    centrality: nx function (default = None)
        nx function to calculate the "mass" of each node
    gravity: int (default = 6)
        the amount to increase the gravity param in forces calculations

    Returns
    -------
    pos : list[float]
        List with the positions of all of the nodes

     Example
    >>> g: nx.Graph = nx.random_tree(70, 1)
    >>> pos = force_directed(g, 1, iterations=1000)
    """
    import numpy as np
    A = nx.to_numpy_array(G)
    k = math.sqrt(1 / len(A))
    if seed is not None:
        logger.info(f"Seed for random position was given: {seed}")
        np.random.seed(seed)
    else:
        logger.info(f"No seed for random position was given")
    logger.info(f"Generating random starting position")
    pos = np.asarray(np.random.rand(len(A), 2))
    logger.info(f'{pos}')
    I = np.zeros(shape=(2, len(A)), dtype=float)
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    gamma_t = 0
    if centrality is None:
        logger.info(
            f"No Centrality type to classify mass was given, therefore the algorithm will use nx.closeness_centrality")
        mass = [v for v in nx.closeness_centrality(G).values()]
    else:
        logger.info(
            f"No Centrality type to classify mass was given, therefore the algorithm will use nx.{centrality}")
        mass = [v for v in centrality(G).values()]
    center = (np.sum(pos, axis=0) / len(pos))

    logger.info(f'Starting iterations: {iterations}, or until gravity force is {gravity*20}')
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
        if gamma_t > gravity * 20:
            break
        if (np.linalg.norm(delta_pos) / len(A)) < threshold:
            threshold /= 3
            # break
            gamma_t += gravity * round(iteration / 200)
            logger.info(f'threshold reached upping gravity force to: {gamma_t}')
        iteration += 1
    logger.info(f'finished calculating positions of graph')
    return pos


def in_hull(point, hull, tolerance=1e-12):
    logger.info("This function checks if a given point is in a given convex hull")
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def convex_pos(hull):
    logger.info("This function computes a new point outside the convex hull to circle the convex hull")
    tmp_pos = []
    center = np.divide(np.sum(hull.points, axis=0), len(hull.points))
    for x in hull.points:
        logger.info("computing linear equation from convex point to the center of the convex")
        m = (x[1] - center[1]) / (x[0] - center[0])
        b = center[1] - m * center[0]
        logger.info("computing a point which has 0.008 distance from the given point and is on the linear equation")
        dist = 0.008
        x0 = x[0] - dist * (math.sqrt(1 / (1 + m ** 2)))
        logger.info("adding the point which is *not* inside the convex hull")
        if not in_hull((x0, m * x0 + b), hull):
            tmp_pos.append((x0, m * x0 + b))
        else:
            x0 = x[0] + dist * (math.sqrt(1 / (1 + m ** 2)))
            tmp_pos.append((x0, m * x0 + b))
    return ConvexHull(tmp_pos)
    # return np.array(tmp_pos)


def angle_between(x0, x1, y0, y1):
    logger.info("This function returns the angle between 2 points")
    xDiff = x1 - x0
    yDiff = y1 - y0
    return math.degrees(math.atan2(yDiff, xDiff))


def random_color():
    logger.info("This function returns a random not too bright color")
    import random
    red = random.random()
    green = random.random()
    blue = random.random()
    if red < 0.1 or green < 0.1 or blue < 0.1:
        return random_color()
    return [red, green, blue]


# @nx.not_implemented_for("directed")
def force_directed_hyper_graphs_using_social_and_gravity_scaling(G: hypergraph_layout.hypergraph,
                                                                 iterations=50, threshold=70e-4, centrality=None,
                                                                 graph_type=None, gravity=6, seed=None):
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

    iterations : int  optional (default=50)
        Maximum number of iterations taken

    threshold: float optional (default = 1e-4)
        Threshold for relative error in node position changes.
        The iteration stops if the error is below this threshold.

    centrality: int optional (default=0)
        Centrality type for the Social gravity field used in the algorithm.

    graph_type: int optional (default=0)
        Graph type for choosing type of conversion from hyper-graph to graph (cycle/wheel/star/complete)

    gravity: int optional (default=6)
        is responsible for the amount of gravity for the pos generation

    seed: int optional (default=None)
        used in generating starting position for nodes in graph



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

    Example
    >>>    E1 = hyperedge([1, 2, 3, 4])
    >>> E2 = hyperedge([5, 6])
    >>> E3 = hyperedge([1, 3, 4])
    >>> E4 = hyperedge([1])
    >>> G = hypergraph([1, 2, 3, 4, 5, 6], [E1, E2,  E4])
    >>> force_directed_hyper_graphs_using_social_and_gravity_scaling(G, 20, graph_type=hypergraph_layout.star_algorithm, seed=1)

    """
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from scipy.interpolate import splprep
    from scipy.interpolate import splev
    from matplotlib.patches import Ellipse
    import math


    if graph_type is None:
        graph_type = hypergraph_layout.complete_algorithm
    logger.info(f'graph type to convert hyper-graph to: {graph_type}')
    g: nx.Graph = graph_type(G)
    logger.info(f'generated graph:'
                f'nodes: {g.nodes}'
                f'edges: {g.edges}')

    pos = force_directed(G=g, seed=seed, iterations=iterations, threshold=threshold, centrality=centrality,
                         gravity=gravity)
    logger.info(f'positions of nodes: {pos}')
    if graph_type is hypergraph_layout.star_algorithm or graph_type is hypergraph_layout.wheel_algorithm:
        logger.info(f'removing addon central nodes')
        pos = pos[:len(pos) - len(G.hyperedges)]
    fig, ax = plt.subplots()
    ax.scatter(pos[:, 0], pos[:, 1], zorder=2)
    logger.info(f'generating the visual plot for the graph')
    for ei in G.hyperedges:
        logger.info(f'calculating convex hull for hyper-edge: {ei.vertices}')
        indexes = []
        for v in ei.vertices:
            indexes.append(np.where(G.vertices == v)[0][0])
        if len(indexes) >= 3:
            hull = ConvexHull(pos[indexes])
            logger.info("Smoothing the drawing")
            new_hull = convex_pos(hull)
            order = get_points_order(new_hull)
            tmp_pos = new_hull.points[order]
            tck, u = splprep(tmp_pos.T, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)

            ax.plot(x_new, y_new, color=random_color(), zorder=0)
        elif len(indexes) == 2:
            x0 = pos[indexes][0][0]
            y0 = pos[indexes][0][1]
            x1 = pos[indexes][1][0]
            y1 = pos[indexes][1][1]
            width = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) + 0.03
            center = ((x0 + x1) / 2, (y0 + y1) / 2)
            angle = angle_between(x0, x1, y0, y1)
            ellipse = Ellipse(center, 0.020, width, angle=angle + 90, fill=False, color=random_color())
            ax.set_aspect(1)
            ax.add_artist(ellipse)
        elif len(indexes) == 1:
            draw_circle = plt.Circle((pos[indexes][0][0], pos[indexes][0][1]), 0.010, fill=False, color=random_color())
            ax.set_aspect(1)
            ax.add_artist(draw_circle)

            # plt.show()
    for i, txt in enumerate(G.vertices):
        ax.annotate(txt, pos[i], color='blue')
    plt.show()
    return pos
