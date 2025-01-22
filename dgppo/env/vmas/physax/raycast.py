import jax.numpy as jnp

from physax.entity import Entity
from physax.jax_types import Pos2, Vec2
from physax.shapes import Box
from physax.vmas_utils import rotate_vector


def cast_ray_to_box(
    box: Entity,
    ray_origin: Pos2,
    ray_direction: Vec2,
    max_range: float,
):
    """
    Inspired from https://tavianator.com/2011/ray_box.html
    Computes distance of ray originating from pos at angle to a box and sets distance to
    max_range if there is no intersection.
    """
    assert ray_origin.naxis == 2 and ray_direction.naxis == 1
    assert ray_origin.shape[0] == ray_direction.shape[0]
    assert isinstance(box.shape, Box)

    pos_origin = ray_origin - box.state.pos
    pos_aabb = rotate_vector(pos_origin, -box.state.rot)
    ray_dir_world = jnp.stack([jnp.cos(ray_direction), jnp.sin(ray_direction)], axis=-1)
    ray_dir_aabb = rotate_vector(ray_dir_world, -box.state.rot)

    tx1 = (-box.shape.length / 2 - pos_aabb[:, 0]) / ray_dir_aabb[:, 0]
    tx2 = (box.shape.length / 2 - pos_aabb[:, 0]) / ray_dir_aabb[:, 0]
    tx = jnp.stack([tx1, tx2], axis=-1)
    tmin, _ = jnp.min(tx, axis=-1)
    tmax, _ = jnp.max(tx, axis=-1)

    ty1 = (-box.shape.width / 2 - pos_aabb[:, 1]) / ray_dir_aabb[:, 1]
    ty2 = (box.shape.width / 2 - pos_aabb[:, 1]) / ray_dir_aabb[:, 1]
    ty = jnp.stack([ty1, ty2], axis=-1)
    tymin, _ = jnp.min(ty, axis=-1)
    tymax, _ = jnp.max(ty, axis=-1)
    tmin, _ = jnp.max(jnp.stack([tmin, tymin], axis=-1), axis=-1)
    tmax, _ = jnp.min(jnp.stack([tmax, tymax], axis=-1), axis=-1)

    intersect_aabb = tmin.unsqueeze(1) * ray_dir_aabb + pos_aabb
    intersect_world = rotate_vector(intersect_aabb, box.state.rot) + box.state.pos

    collision = (tmax >= tmin) & (tmin > 0.0)
    dist = jnp.linalg.norm(ray_origin - intersect_world, axis=1)
    dist[~collision] = max_range
    return dist
