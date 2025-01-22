import jax.numpy as jnp

from .jax_types import RealScalar, Vec1, Vec2


def clamp_with_norm(tensor: jnp.ndarray, max_norm: float):
    norm = jnp.linalg.vector_norm(tensor, axis=-1, keepdims=True)
    new_tensor = (tensor / norm) * max_norm
    tensor = jnp.where(norm > max_norm, new_tensor, tensor)
    return tensor


def rotate_vector(vector: Vec2, angle: RealScalar):
    if len(angle.shape) == len(vector.shape):
        angle = angle.squeeze(-1)

    assert vector.shape[:-1] == angle.shape
    assert vector.shape[-1] == 2

    cos = jnp.cos(angle)
    sin = jnp.sin(angle)
    return jnp.stack(
        [
            vector[..., 0] * cos - vector[..., 1] * sin,
            vector[..., 0] * sin + vector[..., 1] * cos,
        ],
        axis=-1,
    )


def cross(vector_a: Vec2, vector_b: Vec2) -> Vec1:
    return (vector_a[..., 0] * vector_b[..., 1] - vector_a[..., 1] * vector_b[..., 0])[..., None]


def compute_torque(f: Vec2, r: Vec2) -> Vec1:
    return cross(r, f)
