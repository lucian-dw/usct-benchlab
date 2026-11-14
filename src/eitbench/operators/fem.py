"""Original 2-D P1 FEM, CEM/continuum solvers, and discrete adjoint gradients.

Coordinates: m; conductivity: S/m; thickness: m; current: A; voltage: V;
physical contact impedance: ohm m^2. No point-electrode approximation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay


@dataclass
class Mesh:
    nodes: np.ndarray
    triangles: np.ndarray
    boundary_edges: np.ndarray
    electrode_ids: np.ndarray

    def __post_init__(self):
        for name in ("triangles", "boundary_edges", "electrode_ids"):
            if np.asarray(getattr(self, name)).dtype.kind not in "iu":
                raise ValueError(name + " must contain integer indices")
        if np.iscomplexobj(self.nodes):
            raise ValueError("real node coordinates required")
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.triangles = np.asarray(self.triangles, dtype=np.int64)
        self.boundary_edges = np.asarray(self.boundary_edges, dtype=np.int64)
        self.electrode_ids = np.asarray(self.electrode_ids, dtype=np.int64)
        if (
            self.nodes.ndim != 2
            or self.nodes.shape[1] != 2
            or not np.isfinite(self.nodes).all()
        ):
            raise ValueError("nodes must be finite [N,2] coordinates")
        if (
            self.triangles.ndim != 2
            or self.triangles.shape[1] != 3
            or self.triangles.size == 0
        ):
            raise ValueError("triangles must be nonempty [T,3] indices")
        if self.triangles.min() < 0 or self.triangles.max() >= len(self.nodes):
            raise ValueError("invalid triangle indices")
        if (
            self.boundary_edges.ndim != 2
            or self.boundary_edges.shape[1] != 2
            or not self.boundary_edges.size
            or self.electrode_ids.shape != (len(self.boundary_edges),)
            or self.boundary_edges.min() < 0
            or self.boundary_edges.max() >= len(self.nodes)
        ):
            raise ValueError("invalid boundary edges or electrode labels")
        p = self.nodes[self.triangles]
        det = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
            p[:, 2, 0] - p[:, 0, 0]
        ) * (p[:, 1, 1] - p[:, 0, 1])
        if np.any(np.abs(det) < 1e-16):
            raise ValueError("degenerate triangle")
        self.area = np.abs(det) / 2
        self.grad = (
            np.stack(
                (
                    p[:, [1, 2, 0], 1] - p[:, [2, 0, 1], 1],
                    p[:, [2, 0, 1], 0] - p[:, [1, 2, 0], 0],
                ),
                axis=2,
            )
            / det[:, None, None]
        )
        self.local_k = self.area[:, None, None] * np.einsum(
            "tid,tjd->tij", self.grad, self.grad
        )
        self.rows = np.repeat(self.triangles, 3, axis=1).ravel()
        self.cols = np.tile(self.triangles, (1, 3)).ravel()
        self.length = np.linalg.norm(
            self.nodes[self.boundary_edges[:, 1]]
            - self.nodes[self.boundary_edges[:, 0]],
            axis=1,
        )
        if len(np.unique(np.sort(self.boundary_edges, axis=1), axis=0)) != len(
            self.boundary_edges
        ):
            raise ValueError("duplicate boundary edges")
        if np.any(self.length <= 0):
            raise ValueError("zero-length boundary edge")
        # Verify that supplied edges are exactly the exterior of the triangulation.
        all_edges = np.sort(
            np.concatenate(
                [
                    self.triangles[:, [0, 1]],
                    self.triangles[:, [1, 2]],
                    self.triangles[:, [2, 0]],
                ]
            ),
            axis=1,
        )
        unique, counts = np.unique(all_edges, axis=0, return_counts=True)
        if np.any(counts > 2) or set(map(tuple, unique[counts == 1])) != set(
            map(tuple, np.sort(self.boundary_edges, axis=1))
        ):
            raise ValueError("boundary_edges do not describe the mesh exterior")

    def stiffness(self, sigma, thickness):
        sigma = np.asarray(sigma, dtype=float)
        if (
            sigma.shape != (len(self.nodes),)
            or not np.isfinite(sigma).all()
            or np.any(sigma <= 0)
        ):
            raise ValueError("conductivity must be positive finite [N]")
        values = (
            thickness * sigma[self.triangles].mean(axis=1)[:, None, None] * self.local_k
        )
        return coo_matrix(
            (values.ravel(), (self.rows, self.cols)), shape=(len(sigma), len(sigma))
        ).tocsc()

    def tv(self, m, epsilon):
        from .regularization import smooth_tv

        return smooth_tv(self, m, epsilon)


def disk_mesh(radius=0.14, electrode_width=0.025, electrodes=16, rings=8):
    """Deterministic circle triangulation with exact angular electrode endpoints."""
    if (
        radius <= 0
        or rings < 2
        or electrodes < 4
        or not 0 < electrode_width < 2 * np.pi * radius / electrodes
    ):
        raise ValueError("invalid disk geometry")
    points = [[0.0, 0.0]]
    for j in range(1, rings):
        theta = np.arange(electrodes * j) * 2 * np.pi / (electrodes * j)
        points.extend(
            np.c_[
                radius * j / rings * np.cos(theta), radius * j / rings * np.sin(theta)
            ]
        )
    centers = np.pi / 2 - np.arange(electrodes) * 2 * np.pi / electrodes
    endpoints = (
        centers[:, None] + np.array([-1, 1]) * electrode_width / (2 * radius)
    ).ravel() % (2 * np.pi)
    theta = np.arange(electrodes * rings) * 2 * np.pi / (electrodes * rings)
    # Avoid tiny edges next to exact electrode endpoints.
    distances = np.abs(np.angle(np.exp(1j * (theta[:, None] - endpoints))))
    theta = np.sort(np.r_[theta[np.min(distances, axis=1) > 0.008], endpoints])
    start = len(points)
    points.extend(np.c_[radius * np.cos(theta), radius * np.sin(theta)])
    nodes = np.array(points)
    edges = np.c_[
        np.arange(start, len(nodes)), np.roll(np.arange(start, len(nodes)), -1)
    ]
    mid = np.mean(nodes[edges], axis=1)
    angles = np.arctan2(mid[:, 1], mid[:, 0])
    delta = np.abs(np.angle(np.exp(1j * (angles[:, None] - centers))))
    ids = np.argmin(delta, axis=1)
    ids[np.min(delta, axis=1) > electrode_width / (2 * radius) + 1e-10] = -1
    return Mesh(nodes, Delaunay(nodes).simplices, edges, ids)
