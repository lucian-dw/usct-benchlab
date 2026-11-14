"""Smoothed isotropic P1 FEM TV from the supplied EIT submission."""

import numpy as np


def smooth_tv(mesh, m, epsilon):
    d = np.einsum("ti,tid->td", m[mesh.triangles], mesh.grad)
    den = np.sqrt(np.sum(d * d, axis=1) + epsilon**2)
    value = np.sum(mesh.area * (den - epsilon))
    local = mesh.area[:, None] * np.einsum("td,tid->ti", d / den[:, None], mesh.grad)
    return float(value), np.bincount(
        mesh.triangles.ravel(), weights=local.ravel(), minlength=len(m)
    )
