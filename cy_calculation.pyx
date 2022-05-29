import numpy as np

__all__ = [
    'calculate_for_all_nodes_cy'
]
cpdef calculate_for_all_nodes_cy(A, pos, I, center, gamma_t, k, mass):
    for v in range(len(A)):
        delta = (pos[v] - pos).T
        distance = np.sqrt((delta ** 2).sum(axis=0))
        distance = np.where(distance < 0.01, 0.01, distance)
        Ai = A[v]
        # displacement "force"
        I[:, v] += calculate_position_change_for_vertice_cy(Ai, center, delta, distance, gamma_t, k, mass, pos, v)
    return I

def calculate_position_change_for_vertice_cy(Ai, center, delta, distance, gamma_t, k, mass, pos, v):
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
