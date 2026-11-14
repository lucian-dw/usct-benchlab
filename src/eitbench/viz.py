import numpy as np
from scipy.interpolate import LinearNDInterpolator


def rasterize(mesh, conductivity, size=128):
    low, high = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    x, y = np.meshgrid(
        np.linspace(low[0], high[0], size), np.linspace(high[1], low[1], size)
    )
    image = LinearNDInterpolator(mesh.nodes, conductivity)(x, y)
    mask = np.isfinite(image)
    return np.where(mask, image, 0.0), mask


def save_preview(path, image, mask, background):
    from PIL import Image

    t = np.clip((image - 0.05 * background) / (1.9 * background), 0.0, 1.0)
    rgb = np.stack(
        (np.minimum(2 * t, 1.0), 1.0 - np.abs(2 * t - 1), np.minimum(2 * (1 - t), 1.0)),
        axis=-1,
    )
    rgb[~mask] = 1.0
    Image.fromarray(np.uint8(rgb * 255)).resize((512, 512)).save(path)
