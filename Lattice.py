import numpy as np
from scipy.spatial import Voronoi
from itertools import product
import sys
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt






def compute_convex_hull(points):
    """
    Compute the convex hull of a set of points.
    """
    hull = ConvexHull(points)
    return hull






def plot_convex_hull(hull, ax=None):
    """
    Plot the convex hull of a set of points.

    Args:
        hull (ConvexHull): The convex hull object.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): The 3D axis to plot on. If None, a new figure is created.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Create a Poly3DCollection object for the convex hull
    poly3d = [[hull.points[vertex] for vertex in simplex] for simplex in hull.simplices]
    collection = Poly3DCollection(poly3d, alpha=0.5, linewidths=1, edgecolors='r')
    ax.add_collection3d(collection)

    # Set limits and labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])





def crop_voronoi_to_hull(vor, hull):
    """
    Crop the Voronoi diagram to the convex hull defined by the given points and returns the vertices and edges of the cropped Voronoi diagram.
    
    Args:
        vor (Voronoi): The Voronoi object.
        hull (ConvexHull): The convex hull object defining the clipping region.
        
    Returns:
        tuple: A tuple containing the vertices and edges of the cropped Voronoi diagram.
    """
    def is_inside_hull(point, hull):
        # Check if the point is inside the convex hull defined by the equations of the hull
        return all(np.dot(eq[:-1], point) + eq[-1] <= 1e-6 for eq in hull.equations)

    def intersect_edge_with_hull(p1, p2, hull):
        # Check if the edge between p1 and p2 intersects with the convex hull
        # and return the intersection point if it does
        direction = p2 - p1
        for eq in hull.equations:
            normal = eq[:-1]
            offset = eq[-1]
            denom = np.dot(normal, direction)
            if abs(denom) > 1e-6:  # Avoid division by zero
                t = -(np.dot(normal, p1) + offset) / denom
                if 0 <= t <= 1:
                    intersection = p1 + t * direction
                    if is_inside_hull(intersection, hull):
                        return intersection
        return None

    # Find the edges and crop them to the convex hull if necessary
    cropped_edges = set()
    for ridge in vor.ridge_vertices:
        for i in range(len(ridge)):
            i1 = ridge[i]
            i2 = ridge[(i + 1) % len(ridge)]
            # Vertices are at infinity
            if i1 == -1 or i2 == -1:
                continue
            p1 = tuple(vor.vertices[i1])
            p2 = tuple(vor.vertices[i2])
            # Both vertices are inside the hull
            if is_inside_hull(p1, hull) and is_inside_hull(p2, hull):
                cropped_edges.add((p1, p2) if p1 < p2 else (p2, p1))
            # One vertex is inside the hull and the other is outside the hull
            elif is_inside_hull(p1, hull) or is_inside_hull(p2, hull):
                intersection = intersect_edge_with_hull(np.array(p1), np.array(p2), hull)
                if intersection is not None:
                    intersection = tuple(intersection)
                    if is_inside_hull(p1, hull):
                        cropped_edges.add((p1, intersection) if p1 < intersection else (intersection, p1))
                    else:
                        cropped_edges.add((intersection, p2) if intersection < p2 else (p2, intersection))
            # Both vertices are outside the hull
            else:
                continue

    # Create points and connectivities from the cropped edges
    points = []
    connectivities = []
    point_index_map = {}

    for edge in cropped_edges:
        p1, p2 = edge
        if p1 not in point_index_map:
            point_index_map[p1] = len(points)
            points.append(p1)
        if p2 not in point_index_map:
            point_index_map[p2] = len(points)
            points.append(p2)
        connectivities.append((point_index_map[p1], point_index_map[p2]))

    return points, connectivities





class Lattice:

    def __init__(self, points=None, connectivity=None, file_path=None):
        """
        Create a Lattice object either from points and connectivity or from a file.

        Args:
            points (list or np.ndarray, optional): List or array of points.
            connectivity (list or np.ndarray, optional): List or array of connectivity.
            file_path (str, optional): Path to the file containing points and connectivity.
        """
        if file_path:
            self.points = np.empty((0, 3))
            self.connectivity = np.empty((0, 2), dtype=int)

            with open(file_path, 'r') as file:
                lines = file.readlines()
                reading_points = True
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.lower() == "connectivity":
                        reading_points = False
                        continue
                    if reading_points:
                        self.points = np.vstack([self.points, list(map(float, line.split(',')))])
                    else:
                        self.connectivity = np.vstack([self.connectivity, list(map(int, line.split(',')))])
        else:
            self.points = np.array(points) if points is not None else np.empty((0, 3))
            self.connectivity = np.array(connectivity) if connectivity is not None else np.empty((0, 2), dtype=int)



    def tessellate(self, vectors=None, N=3):
        """
        Tessellate the points and connectivity in the lattice using the specified vectors and multiplicities.

        Args:
            vectors (list of np.ndarray): List of vectors to use for tessellation.
            N (int): Number of tessellations per direction.
        """
        vectors = np.array(vectors) if vectors is not None else np.eye(3)
        tessellated_points = self.points.tolist()
        tessellated_connectivity = []
        # Generate all possible multiplicities trying to keep the original lattice centered
        multiplicities = list(product(range(-((N-1)//2), ((N)//2) + 1), repeat=len(vectors)))
        point_index_map = {tuple(p): i for i, p in enumerate(tessellated_points)}
        
        for mult in multiplicities:
            v = np.zeros(3)
            for i, m in enumerate(mult):
                v += m * vectors[i]
            translated_points = self.points + v
            
            # Map old connectivity to new tessellated points
            offset = len(tessellated_points)
            for tp in translated_points:
                tp_tuple = tuple(tp)
                if tp_tuple not in point_index_map:
                    point_index_map[tp_tuple] = len(tessellated_points)
                    tessellated_points.append(tp.tolist())
            
            for edge in self.connectivity:
                p1 = tuple(self.points[edge[0]] + v)
                p2 = tuple(self.points[edge[1]] + v)
                if p1 in point_index_map and p2 in point_index_map:
                    tessellated_connectivity.append((point_index_map[p1], point_index_map[p2]))
        
        return np.array(tessellated_points), np.array(tessellated_connectivity)

    def compute_dual(self):
        """
        Compute the dual of the lattice using Voronoi tessellation.
        """
        hull = compute_convex_hull(self.points)
        tessellated_points, _ = self.tessellate(N=3)
        vor = Voronoi(tessellated_points)
        dual_points, dual_connectivities = crop_voronoi_to_hull(vor, hull)
        return Lattice(dual_points, dual_connectivities)

    def plot_lattice(self, ax=None, c='k', s=50):
        """
        Plot the lattice points and connectivity.

        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot, optional): The 3D axis to plot on. If None, a new figure is created.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Plot points
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c=c, marker='o', s=s)

        # Plot edges
        for edge in self.connectivity:
            p1 = self.points[edge[0]]
            p2 = self.points[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=c)

        # Set axis labels
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])


lattice = Lattice(file_path='unit-cells/bcc.dat')
lattice.plot_lattice(c='b')
dual_lattice = lattice.compute_dual()
dual_lattice.plot_lattice(ax=plt.gca(), c='r')
plt.show()
