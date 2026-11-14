"""CEM and continuum P1 FEM; exact discrete conductivity adjoint."""

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, bmat
from scipy.sparse.linalg import splu


class ForwardModel:
    def __init__(
        self,
        mesh,
        currents,
        measurement,
        thickness=0.07,
        model="cem",
        contact=0.001,
        ledger=None,
    ):
        self.ledger = ledger
        self.mesh, self.model, self.thickness = mesh, model, float(thickness)
        self.currents, self.measurement = np.asarray(currents, float), np.asarray(
            measurement, float
        )
        if self.currents.ndim != 2 or self.measurement.ndim != 2:
            raise ValueError("currents [L,P] and measurement [Q,L] must be matrices")
        self.n, self.l = len(mesh.nodes), self.currents.shape[0]
        if (
            self.measurement.shape[1] != self.l
            or not np.isfinite(self.currents).all()
            or not np.isfinite(self.measurement).all()
        ):
            raise ValueError("incompatible or nonfinite protocol")
        if not np.allclose(self.currents.sum(axis=0), 0, atol=1e-12, rtol=0):
            raise ValueError("each current pattern must sum to zero")
        if not np.allclose(self.measurement.sum(axis=1), 0, atol=1e-12, rtol=0):
            raise ValueError("voltage measurements must be gauge-invariant differences")
        if (
            model not in ("cem", "continuum")
            or not np.isfinite(thickness)
            or thickness <= 0
        ):
            raise ValueError("invalid model or thickness")
        if np.any(mesh.electrode_ids < -1) or np.any(mesh.electrode_ids >= self.l):
            raise ValueError("invalid electrode label")
        self.contact = np.broadcast_to(np.asarray(contact, float), (self.l,)).copy()
        if not np.isfinite(self.contact).all() or np.any(self.contact <= 0):
            raise ValueError("positive contact impedances required")
        self.integrals = np.zeros((self.n, self.l))
        self.lengths = np.zeros(self.l)
        self.gauge = np.zeros(self.n)
        for edge, eid, length in zip(
            mesh.boundary_edges, mesh.electrode_ids, mesh.length
        ):
            self.gauge[edge] += length / 2
            if eid >= 0:
                self.integrals[edge, eid] += length / 2
                self.lengths[eid] += length
        if np.any(self.lengths <= 0):
            raise ValueError("every electrode needs at least one boundary edge")
        self.average = self.integrals / self.lengths
        if model == "cem":
            self.size = self.n + self.l + 1
            self.observation = np.zeros((len(measurement), self.size))
            self.observation[:, self.n : self.n + self.l] = self.measurement
            self.rhs = np.zeros((self.size, self.currents.shape[1]))
            self.rhs[self.n : self.n + self.l] = self.currents
        else:
            self.size = self.n + 1
            self.observation = np.zeros((len(measurement), self.size))
            self.observation[:, : self.n] = self.measurement @ self.average.T
            self.rhs = np.zeros((self.size, self.currents.shape[1]))
            self.rhs[: self.n] = self.average @ self.currents

    def matrix(self, sigma):
        k = self.mesh.stiffness(sigma, self.thickness)
        if self.model == "continuum":
            c = csc_matrix(self.gauge[:, None])
            return bmat([[k, c], [c.T, None]], format="csc")
        rows, cols, vals = [], [], []
        for edge, eid, length in zip(
            self.mesh.boundary_edges, self.mesh.electrode_ids, self.mesh.length
        ):
            if eid < 0:
                continue
            local = (
                np.array([[2, 1, -3], [1, 2, -3], [-3, -3, 6]], float)
                * length
                * self.thickness
                / (6 * self.contact[eid])
            )
            indices = np.r_[edge, self.n + eid]
            rows.extend(np.repeat(indices, 3))
            cols.extend(np.tile(indices, 3))
            vals.extend(local.ravel())
        contact = coo_matrix(
            (vals, (rows, cols)), shape=(self.n + self.l, self.n + self.l)
        ).tocsc()
        base = (
            bmat([[k, None], [None, csc_matrix((self.l, self.l))]], format="csc")
            + contact
        )
        c = csc_matrix(np.r_[np.zeros(self.n), np.ones(self.l)][:, None])
        return bmat([[base, c], [c.T, None]], format="csc")

    def solve(self, sigma):
        if self.ledger is not None:
            self.ledger.charge("forward")
        lu = splu(self.matrix(sigma))
        state = lu.solve(self.rhs)
        if not np.isfinite(state).all():
            raise FloatingPointError("nonfinite FEM state")
        return self.observation @ state, state, lu

    def adjoint_gradient(self, state, lu, voltage_derivative):
        if self.ledger is not None:
            self.ledger.charge("adjoint")
        # A^T p = H^T dJ/dV; dJ/dsigma_i = -p^T (dA/dsigma_i) u.
        adjoint = lu.solve(self.observation.T @ voltage_derivative, trans="T")
        tri = self.mesh.triangles
        gu = np.einsum("tip,tid->tdp", state[: self.n][tri], self.mesh.grad)
        gp = np.einsum("tip,tid->tdp", adjoint[: self.n][tri], self.mesh.grad)
        element = -self.thickness * self.mesh.area * np.sum(gu * gp, axis=(1, 2)) / 3
        gradient = np.bincount(
            tri.ravel(), weights=np.repeat(element, 3), minlength=self.n
        )
        return gradient, adjoint

    def contact_log_gradient(self, state, adjoint):
        result = np.zeros(self.l)
        for edge, eid, length in zip(
            self.mesh.boundary_edges, self.mesh.electrode_ids, self.mesh.length
        ):
            if eid < 0:
                continue
            a = state[edge] - state[self.n + eid]
            b = adjoint[edge] - adjoint[self.n + eid]
            mass = length * np.array([[2, 1], [1, 2]]) / 6
            # derivative wrt log(z): dA/dlog(z) = -A_contact.
            result[eid] += self.thickness / self.contact[eid] * np.sum(b * (mass @ a))
        return result

    def continuum_flux(self, sigma, nodal_flux):
        """Solve arbitrary continuous P1 boundary sheet flux (A/m), not electrode data."""
        if self.model != "continuum":
            raise ValueError("continuum model required")
        flux = np.asarray(nodal_flux, float)
        if flux.shape != (self.n,) or not np.isfinite(flux).all():
            raise ValueError("nodal_flux must be finite [N]")
        load = np.zeros(self.n + 1)
        for edge, length in zip(self.mesh.boundary_edges, self.mesh.length):
            load[edge] += length / 6 * np.array([[2, 1], [1, 2]]) @ flux[edge]
        if abs(load.sum()) > 1e-10 * max(1.0, np.abs(load).sum()):
            raise ValueError("boundary flux must have zero integral")
        if self.ledger is not None:
            self.ledger.charge("forward")
        return splu(self.matrix(sigma)).solve(load)[: self.n]
