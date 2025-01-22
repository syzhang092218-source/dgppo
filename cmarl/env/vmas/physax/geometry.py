import jax.numpy as jnp
import numpy as np

from .jax_utils import jax_unbind
from .vmas_utils import cross


def get_closest_point_line(
    line_pos: jnp.ndarray,
    line_rot: jnp.ndarray,
    line_length,
    test_point_pos,
    limit_to_line_length: bool = True,
):
    assert line_rot.shape[-1] == 1

    # Rotate it by the angle of the line
    # (..., 2)
    rotated_vector = jnp.concatenate([jnp.cos(line_rot), jnp.sin(line_rot)], axis=-1)

    # Get distance between line and sphere
    # (..., 2)
    delta_pos = line_pos - test_point_pos
    # Dot product of distance and line vector
    # (..., 1)
    dot_p = (delta_pos * rotated_vector).sum(-1, keepdims=True)
    # Coordinates of the closes point
    sign = jnp.sign(dot_p)
    if limit_to_line_length:
        distance_from_line_center = jnp.minimum(jnp.abs(dot_p), (line_length / 2)[..., None])
    else:
        distance_from_line_center = jnp.abs(dot_p)
    closest_point = line_pos - sign * distance_from_line_center * rotated_vector
    return closest_point


def get_closest_point_box(
    box_pos: jnp.ndarray,
    box_rot: jnp.ndarray,
    box_width: jnp.ndarray,
    box_length: np.ndarray,
    test_point_pos: jnp.ndarray,
):
    closest_points = get_all_points_box(box_pos, box_rot, box_width, box_length, test_point_pos)
    closest_point = jnp.full(box_pos.shape, np.inf)
    distance = jnp.full(box_pos.shape[:-1], np.inf)
    for p in closest_points:
        d = jnp.linalg.vector_norm(test_point_pos - p, axis=-1)
        is_closest = d < distance
        closest_point = jnp.where(is_closest[..., None], p, closest_point)
        distance = jnp.where(is_closest, d, distance)

    return closest_point


def get_all_points_box(
    box_pos: jnp.ndarray,
    box_rot: jnp.ndarray,
    box_width: np.ndarray,
    box_length: np.ndarray,
    test_point_pos: jnp.ndarray,
):
    lines_pos, lines_rot, lines_length = get_all_lines_box(box_pos, box_rot, box_width, box_length)

    closest_points = jax_unbind(
        get_closest_point_line(
            lines_pos,
            lines_rot,
            lines_length,
            test_point_pos,
        ),
        axis=0,
    )

    return closest_points


def get_all_lines_box(box_pos: jnp.ndarray, box_rot: jnp.ndarray, box_width: np.ndarray, box_length: np.ndarray):
    # rot: (..., 1)
    # (..., n_box, 2) Rotate normal vector by the angle of the box
    rotated_vector = jnp.concatenate([jnp.cos(box_rot), jnp.sin(box_rot)], axis=-1)
    rot_2 = box_rot + np.pi / 2
    rotated_vector2 = jnp.concatenate([jnp.cos(rot_2), jnp.sin(rot_2)], axis=-1)

    expanded_half_box_length = box_length / 2
    expanded_half_box_width = box_width / 2

    # Middle points of the sides
    p1 = box_pos + rotated_vector * expanded_half_box_length[..., None]
    p2 = box_pos - rotated_vector * expanded_half_box_length[..., None]
    p3 = box_pos + rotated_vector2 * expanded_half_box_width[..., None]
    p4 = box_pos - rotated_vector2 * expanded_half_box_width[..., None]

    ps = []
    rots = []
    lengths = []
    for i, p in enumerate([p1, p2, p3, p4]):
        ps.append(p)
        rots.append(box_rot + np.pi / 2 if i <= 1 else box_rot)
        lengths.append(box_width if i <= 1 else box_length)

    return jnp.stack(ps, axis=0), jnp.stack(rots, axis=0), jnp.stack(lengths, axis=0)


def get_closest_line_box(
    box_pos: jnp.ndarray,
    box_rot: jnp.ndarray,
    box_width: np.ndarray,
    box_length: np.ndarray,
    line_pos: jnp.ndarray,
    line_rot: jnp.ndarray,
    line_length: jnp.ndarray,
):
    lines_pos, lines_rot, lines_length = get_all_lines_box(box_pos, box_rot, box_width, box_length)

    closest_point_1 = jnp.full(box_pos.shape, np.inf)
    closest_point_2 = jnp.full(box_pos.shape, np.inf)
    distance = jnp.full(box_pos.shape[:-1], np.inf)

    ps_box, ps_line = get_closest_points_line_line(
        lines_pos,
        lines_rot,
        lines_length,
        line_pos,
        line_rot,
        line_length,
    )

    for p_box, p_line in zip(ps_box.unbind(0), ps_line.unbind(0)):
        d = jnp.linalg.vector_norm(p_box - p_line, axis=-1)
        is_closest = d < distance
        is_closest_exp = is_closest.unsqueeze(-1).expand(closest_point_1.shape)
        closest_point_1 = jnp.where(is_closest_exp, p_box, closest_point_1)
        closest_point_2 = jnp.where(is_closest_exp, p_line, closest_point_2)
        distance = jnp.where(is_closest, d, distance)
    return closest_point_1, closest_point_2


def get_closest_points_line_line(
    line_pos: jnp.ndarray,
    line_rot: jnp.ndarray,
    line_length: jnp.ndarray,
    line2_pos: jnp.ndarray,
    line2_rot: jnp.ndarray,
    line2_length: jnp.ndarray,
):
    points_a, points_b = get_line_extrema(
        jnp.stack([line_pos, line2_pos], axis=0),
        jnp.stack([line_rot, line2_rot], axis=0),
        jnp.stack([line_length, line2_length], axis=0),
    )
    point_a1, point_b1 = jax_unbind(points_a, 0)
    point_a2, point_b2 = jax_unbind(points_b, 0)

    # (..., 2), (..., )
    point_i, d_i = get_intersection_point_line_line(point_a1, point_a2, point_b1, point_b2)

    (
        point_a1_line_b,
        point_a2_line_b,
        point_b1_line_a,
        point_b2_line_a,
    ) = get_closest_point_line(
        jnp.stack([line2_pos, line2_pos, line_pos, line_pos], axis=0),
        jnp.stack([line2_rot, line2_rot, line_rot, line_rot], axis=0),
        jnp.stack([line2_length, line2_length, line_length, line_length], axis=0),
        jnp.stack([point_a1, point_a2, point_b1, point_b2], axis=0),
    )

    point_pairs = (
        (point_a1, point_a1_line_b),
        (point_a2, point_a2_line_b),
        (point_b1_line_a, point_b1),
        (point_b2_line_a, point_b2),
    )

    closest_point_1 = jnp.full(line_pos.shape, np.inf)
    closest_point_2 = jnp.full(line_pos.shape, np.inf)
    min_distance = jnp.full(line_pos.shape[:-1], np.inf)
    for p1, p2 in point_pairs:
        d = jnp.linalg.vector_norm(p1 - p2, axis=-1)
        is_closest = d < min_distance
        is_closest_exp = is_closest.unsqueeze(-1).expand(p1.shape)
        closest_point_1 = jnp.where(is_closest_exp, p1, closest_point_1)
        closest_point_2 = jnp.where(is_closest_exp, p2, closest_point_2)
        min_distance = jnp.where(is_closest, d, min_distance)

    cond = (d_i == 0)[..., None]
    closest_point_1 = jnp.where(cond, point_i, closest_point_1)
    closest_point_2 = jnp.where(cond, point_i, closest_point_2)

    assert closest_point_1.shpae == closest_point_2.shape == line_pos.shape

    return closest_point_1, closest_point_2


def get_line_extrema(line_pos: jnp.ndarray, line_rot: jnp.ndarray, line_length: jnp.ndarray):
    line_length = line_length.view(line_rot.shape)
    x = (line_length / 2) * jnp.cos(line_rot)
    y = (line_length / 2) * jnp.sin(line_rot)
    xy = jnp.concatenate([x, y], axis=-1)

    point_a = line_pos + xy
    point_b = line_pos - xy

    return point_a, point_b


def get_intersection_point_line_line(
    point_a1: jnp.ndarray, point_a2: jnp.ndarray, point_b1: jnp.ndarray, point_b2: jnp.ndarray
):
    """
    Taken from:
    https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    """

    # point_a1: (..., 2)

    r = point_a2 - point_a1
    s = point_b2 - point_b1
    p = point_a1
    q = point_b1
    # (..., 1)
    cross_q_minus_p_r = cross(q - p, r)
    cross_q_minus_p_s = cross(q - p, s)
    cross_r_s = cross(r, s)
    # (..., 1)
    u = cross_q_minus_p_r / cross_r_s
    t = cross_q_minus_p_s / cross_r_s
    t_in_range = (0 <= t) * (t <= 1)
    u_in_range = (0 <= u) * (u <= 1)

    # (..., 1)
    cross_r_s_is_zero = cross_r_s == 0

    # (..., )
    distance = jnp.full(point_a1.shape[:-1], np.inf)
    # (..., 2)
    point = jnp.full(point_a1.shape, np.inf)

    # (..., 1)
    condition = ~cross_r_s_is_zero * u_in_range * t_in_range
    condition_exp = condition

    # (..., 2)
    point = jnp.where(condition_exp, p + t * r, point)
    # (..., )
    distance = jnp.where(condition.squeeze(-1), 0.0, distance)

    assert point.shape == point_a1.shape
    assert distance.shape == point_a1.shape[:-1]

    return point, distance
