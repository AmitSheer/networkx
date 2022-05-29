'''First artical: Force-Directed Graph Drawing Using Social Gravity and Scaling
authots: Michael J. Bannister, David Eppstein, Michael T. Goodrich and Lowell Trott
link: https://arxiv.org/pdf/1209.0748.pdf

Second artical: Hypergraph Drawing by Force-Directed Placement
authots: Naheed Anjum Arafat and Stephane Bressan
link: https://www.researchgate.net/publication/318823299_Hypergraph_Drawing_by_Force-Directed_Placement

Our names: Amit Sheer Cohen and Neta Roth'''

import math
import threading

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


def force_directed(G: nx.Graph, seed: int, iterations: int = 50, threshold=70e-4, centrality=None, gravity: int = 6,
                   gravity_multiplier: float = 20., dthreshold: float = 3):
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
    gravity_multiplier: float (default=20)
        the multiplier of the gravitational force in the step function
    dthreshold: float (default=3.)
        how much to divide the threshold when reaching the current threshold

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
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    gamma_t = 0
    mass = get_mass(G, centrality)

    center = (np.sum(pos, axis=0) / len(pos))
    logger.info(f'Starting iterations: {iterations}, or until gravity force is {gravity * 20}')
    for iteration in range(iterations):
        I *= 0
        for v in range(len(A)):
            delta = (pos[v] - pos).T
            distance = np.sqrt((delta ** 2).sum(axis=0))
            distance = np.where(distance < 0.01, 0.01, distance)
            Ai = A[v]
            # displacement "force"
            I[:, v] += calculate_position_change_for_vertice(Ai, center, delta, distance, gamma_t, k, mass, pos, v)
        length = np.sqrt((I ** 2).sum(axis=0))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = (I * t / length).T
        pos += delta_pos
        # cool temperature
        t -= dt

        if gamma_t > gravity * gravity_multiplier:
            break
        if (np.linalg.norm(delta_pos) / len(A)) < threshold:
            threshold /= dthreshold
            gamma_t += gravity * round(iteration / 200)
            logger.info(f'threshold reached upping gravity force to: {gamma_t}')
        iteration += 1
    logger.info(f'finished calculating positions of graph')
    return pos


def calculate_position_change_for_vertice(Ai, center, delta, distance, gamma_t, k, mass, pos, v):
    """

    Parameters
    ----------
    Ai: matrix
        adjacency's matrix for node v
    center:
        the position of the relative center

    delta:
        matrix of the vector from v to all other vertices
    distance: matrix
        normalized distance from v to all other nodes in graph
    gamma_t: float
        the gravity force
    k: float
        area and minimum distance between two nodes
    mass:
        the mass/centrality of v
    pos
    v: int
    id of v

    Returns
    -------
    the calculations of the change in position
    """
    return (
                   delta * (k * k / distance ** 2 - Ai * distance / k)
           ).sum(axis=1) + gamma_t * mass[v] * (center - pos[v])


def get_mass(G, centrality):
    """

    Parameters
    ----------
    G: nx.Graph
        the graph to run the centrality algorithm on
    centrality:
        nx function about what kind of centrality to use
        nx.closeness_centrality, nx.degree_centrality, nx.betweenness_centrality

    Returns
    -------
    mass: np.array
        returns an array with the values of the centrality
    """
    if centrality is None:
        logger.info(
            f"No Centrality type to classify mass was given, therefore the algorithm will use nx.closeness_centrality")
        mass = [v for v in nx.closeness_centrality(G).values()]
    else:
        logger.info(
            f"No Centrality type to classify mass was given, therefore the algorithm will use nx.{centrality}")
        mass = [v for v in centrality(G).values()]
    return mass


def in_hull(point, hull, tolerance=1e-12):
    """
       Parameters
       ----------
       point:
           the point function will calculate if inside the convex hull
       hull:
           the hull the point is asked about
       tolerance:
           tolerance for correctness

       Returns
       -------
       mass: bool
            if the point inside the convex hull- True, otherwise- False
       """
    logger.info("This function checks if a given point is in a given convex hull")
    # taken from
    # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    # In words, a point is in the hull if and only if for every equation (describing the facets) the dot product between
    # the point and the normal vector (eq[:-1]) plus the offset (eq[-1]) is less than or equal to zero.
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def convex_pos(hull, proportion=400):
    """
       Parameters
       ----------
       hull:
           the hull which the function calculates it's smoothing positions
       proportion:
           The parameter that determines how much to divide the canvas
           (to get the distance from the real position of the point to its new position)
       Returns
       -------
       hull:
           the hull with the new positions
       """
    logger.info("This function computes a new point outside the convex hull to circle the convex hull")
    tmp_pos = []
    center = np.divide(np.sum(hull.points, axis=0), len(hull.points))
    for x in hull.points:
        logger.info("computing linear equation from convex point to the center of the convex")
        m = (x[1] - center[1]) / (x[0] - center[0])
        b = center[1] - m * center[0]
        logger.info("computing a point which has proportional"
                    " distance from the given point and is on the linear equation")

        dist = plt.gcf().get_size_inches()[0] / proportion
        x0 = x[0] - dist * (math.sqrt(1 / (1 + m ** 2)))
        logger.info("adding the point which is *not* inside the convex hull")
        if not in_hull((x0, m * x0 + b), hull):
            tmp_pos.append((x0, m * x0 + b))
        else:
            x0 = x[0] + dist * (math.sqrt(1 / (1 + m ** 2)))
            tmp_pos.append((x0, m * x0 + b))
    return ConvexHull(tmp_pos)


def angle_between(x0, x1, y0, y1):
    """
         Parameters
         ----------
         x0:
             x coordinate of the first point
         y0:
             y coordinate of the first point
         x1
             x coordinate of the second point
         y0:
             y coordinate of the second point
         Returns
         -------
         int:
             degree between 2 points
         """
    logger.info("This function returns the angle between 2 points")
    xDiff = x1 - x0
    yDiff = y1 - y0
    return math.degrees(math.atan2(yDiff, xDiff))


def random_color(brightness_threshold=0.2):
    """
       Parameters
       ----------
       brightness_threshold:
           determines the brighter a color may be
       Returns
       -------
       int:
           random color (with RBG values > brightness_threshold)
       """
    logger.info("This function returns a random not too bright color")
    import random
    red = random.random()
    green = random.random()
    blue = random.random()
    if red < brightness_threshold or green < brightness_threshold or blue < brightness_threshold:
        return random_color()
    return [red, green, blue]


def draw_simplex(pos, indexes, ax):
    from scipy.spatial import ConvexHull
    from scipy.interpolate import splprep
    from scipy.interpolate import splev
    hull = ConvexHull(pos[indexes])
    logger.info("Getting the edge positions")
    new_hull = convex_pos(hull)
    logger.info("Getting the order of the edges")
    order = get_points_order(new_hull)
    tmp_pos = new_hull.points[order]
    logger.info("Smoothing the drawing")
    # taken from https://stackoverflow.com/questions/31464345/fitting-a-closed-curve-to-a-set-of-points
    tck, u = splprep(tmp_pos.T, u=None, s=0.0, per=1)
    smooting_param = 1000
    u_new = np.linspace(u.min(), u.max(), smooting_param)
    x_new, y_new = splev(u_new, tck, der=0)
    ax.plot(x_new, y_new, color=random_color(), zorder=0)


def draw_ellipse(pos, indexes, ax):
    from matplotlib.patches import Ellipse
    x0 = pos[indexes][0][0]
    y0 = pos[indexes][0][1]
    x1 = pos[indexes][1][0]
    y1 = pos[indexes][1][1]
    dist_from_point = 0.03
    height = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) + dist_from_point
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    angle = angle_between(x0, x1, y0, y1)
    ellipse = Ellipse(center, width=0.020, height=height, angle=angle + 90, fill=False, color=random_color())
    ax.add_artist(ellipse)


def draw_circle(pos, indexes, ax):
    logger.info("Getting the circle size proportional to the canvas size")
    proportion = 200
    size = plt.gcf().get_size_inches()[0] / proportion
    draw_circle = plt.Circle((pos[indexes][0][0], pos[indexes][0][1]), size, fill=False, color=random_color())
    ax.add_artist(draw_circle)


def active_with_threads(G, pos, ax):
    threads = []
    for ei in G.hyperedges:
        logger.info(f'calculating convex hull for hyper-edge: {ei.vertices}')
        indexes = []
        for v in ei.vertices:
            indexes.append(np.where(G.vertices == v)[0][0])
        if len(indexes) >= 3:
            t = threading.Thread(target=draw_simplex, args=[pos, indexes, ax])
            t.start()
            threads.append(t)
        elif len(indexes) == 2:
            t = threading.Thread(target=draw_ellipse, args=[pos, indexes, ax])
            t.start()
            threads.append(t)
        elif len(indexes) == 1:
            t = threading.Thread(target=draw_circle, args=[pos, indexes, ax])
            t.start()
            threads.append(t)
    for t in threads:
        t.join()


def active_without_threads(G, pos, ax):
    for ei in G.hyperedges:
        logger.info(f'calculating convex hull for hyper-edge: {ei.vertices}')
        indexes = []
        for v in ei.vertices:
            indexes.append(np.where(G.vertices == v)[0][0])
        if len(indexes) >= 3:
            draw_simplex(pos, indexes, ax)
        elif len(indexes) == 2:
            draw_ellipse(pos, indexes, ax)
        elif len(indexes) == 1:
            draw_circle(pos, indexes, ax)


# @nx.not_implemented_for("directed")
def force_directed_hyper_graphs_using_social_and_gravity_scaling(G: hypergraph_layout.hypergraph,
                                                                 iterations=50, threshold=70e-4, centrality=None,
                                                                 graph_type=None, gravity=6, seed=None, title=None,
                                                                 with_threads=True):
    """Positions nodes using Fruchterman-Reingold force-directed algorithm combined with Hyper-Graphs and Social and
    Gravitational Forces.

    Receives a Hyper-Graph and changes it to be a normal graph. Then calculates the Value of each node according to
    Social centrality Parameters. Nodes with high value are counted as central nodes in the graph and are more attracted
    to the center of the graph dimension.

    After running the algorithm the pos will be updated to reflect the social and force-directed values of the nodes.


    Parameters
    ----------
    title
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
    import time
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
    logger.info('drawing proportional vertices as circles')
    size = plt.gcf().get_size_inches()[0]
    ax.scatter(pos[:, 0], pos[:, 1], s=size, zorder=2)
    logger.info(f'generating the visual plot for the graph')
    if with_threads:
        active_with_threads(G, pos, ax)
    else:
        active_without_threads(G, pos, ax)
    logger.info("drawing vertices' numbers")
    for i, txt in enumerate(G.vertices):
        ax.annotate(txt, pos[i], color='blue')
    if title is not None:
        plt.title(title)
    plt.show()
    return pos
