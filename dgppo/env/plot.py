import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jax
import pathlib
import functools as ft

from colour import hsl2hex
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import Axes
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import proj3d, Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from typing import List, Optional, Union, Tuple

from ..trainer.data import Rollout
from ..trainer.utils import centered_norm
from ..utils.typing import EdgeIndex, Pos2d, Pos3d, Array
from ..utils.utils import merge01, tree_index, MutablePatchCollection, save_anim
from dgppo.env.obstacle import Cuboid, Sphere, Obstacle, Rectangle


def plot_graph(
        ax: Axes,
        pos: Pos2d,
        radius: Union[float, List[float]],
        color: Union[str, List[str]],
        with_label: Union[bool, List[bool]] = True,
        plot_edge: bool = False,
        edge_index: Optional[EdgeIndex] = None,
        edge_color: Union[str, List[str]] = 'k',
        alpha: float = 1.0,
        obstacle_color: str = '#000000',
) -> Axes:
    if isinstance(radius, float):
        radius = np.ones(pos.shape[0]) * radius
    if isinstance(radius, list):
        radius = np.array(radius)
    if isinstance(color, str):
        color = [color for _ in range(pos.shape[0])]
    if isinstance(with_label, bool):
        with_label = [with_label for _ in range(pos.shape[0])]
    circles = []
    for i in range(pos.shape[0]):
        circles.append(plt.Circle((float(pos[i, 0]), float(pos[i, 1])),
                                  radius=radius[i], color=color[i], clip_on=False, alpha=alpha, linewidth=0.0))
        if with_label[i]:
            ax.text(float(pos[i, 0]), float(pos[i, 1]), f'{i}', size=12, color="k",
                    family="sans-serif", weight="normal", horizontalalignment="center", verticalalignment="center",
                    transform=ax.transData, clip_on=True)
    circles = PatchCollection(circles, match_original=True)
    ax.add_collection(circles)

    if plot_edge and edge_index is not None:
        if isinstance(edge_color, str):
            edge_color = [edge_color for _ in range(edge_index.shape[1])]
        start, end = pos[edge_index[0, :]], pos[edge_index[1, :]]
        direction = (end - start) / jnp.linalg.norm(end - start, axis=1, keepdims=True)
        start = start + direction * radius[edge_index[0, :]][:, None]
        end = end - direction * radius[edge_index[1, :]][:, None]
        widths = (radius[edge_index[0, :]] + radius[edge_index[1, :]]) * 20
        lines = np.stack([start, end], axis=1)
        edges = LineCollection(lines, colors=edge_color, linewidths=widths, alpha=0.5)
        ax.add_collection(edges)
    return ax


def plot_node_3d(ax, pos: Pos3d, r: float, color: str, alpha: float, grid: int = 10) -> Axes:
    u = np.linspace(0, 2 * np.pi, grid)
    v = np.linspace(0, np.pi, grid)
    x = r * np.outer(np.cos(u), np.sin(v)) + pos[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + pos[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)
    return ax


def plot_graph_3d(
        ax,
        pos: Pos3d,
        radius: float,
        color: Union[str, List[str]],
        with_label: bool = True,
        plot_edge: bool = False,
        edge_index: Optional[EdgeIndex] = None,
        edge_color: Union[str, List[str]] = 'k',
        alpha: float = 1.0,
        obstacle_color: str = '#000000',
):
    if isinstance(color, str):
        color = [color for _ in range(pos.shape[0])]
    for i in range(pos.shape[0]):
        plot_node_3d(ax, pos[i], radius, color[i], alpha)
        if with_label:
            ax.text(pos[i, 0], pos[i, 1], pos[i, 2], f'{i}', size=12, color="k", family="sans-serif", weight="normal",
                    horizontalalignment="center", verticalalignment="center")
    if plot_edge:
        if isinstance(edge_color, str):
            edge_color = [edge_color for _ in range(edge_index.shape[1])]
        for i_edge in range(edge_index.shape[1]):
            i = edge_index[0, i_edge]
            j = edge_index[1, i_edge]
            vec = pos[i, :] - pos[j, :]
            x = [pos[i, 0] - 2 * radius * vec[0], pos[j, 0] + 2 * radius * vec[0]]
            y = [pos[i, 1] - 2 * radius * vec[1], pos[j, 1] + 2 * radius * vec[1]]
            z = [pos[i, 2] - 2 * radius * vec[2], pos[j, 2] + 2 * radius * vec[2]]
            ax.plot(x, y, z, linewidth=1.0, color=edge_color[i_edge])
    return ax


def get_BuRd():
    # blue = "#3182bd"
    # blue = hsl2hex([0.57, 0.59, 0.47])
    blue = hsl2hex([0.57, 0.5, 0.55])
    light_blue = hsl2hex([0.5, 1.0, 0.995])

    # Tint it to orange a bit.
    # red = "#de2d26"
    # red = hsl2hex([0.04, 0.74, 0.51])
    red = hsl2hex([0.028, 0.62, 0.59])
    light_red = hsl2hex([0.098, 1.0, 0.995])

    sdf_cm = LinearSegmentedColormap.from_list("SDF", [(0, light_blue), (0.5, blue), (0.5, red), (1, light_red)], N=256)
    return sdf_cm


def get_faces_cuboid(points: Pos3d) -> Array:
    point_id = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]])
    faces = points[point_id]
    return faces


def get_cuboid_collection(
        obstacles: Cuboid, alpha: float = 0.8, linewidth: float = 1.0, edgecolor: str = 'k', facecolor: str = 'r'
) -> Poly3DCollection:
    get_faces_vmap = jax.vmap(get_faces_cuboid)
    cuboid_col = Poly3DCollection(
        merge01(get_faces_vmap(obstacles.points)),
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor
    )
    return cuboid_col


def get_sphere_collection(
        obstacles: Sphere, alpha: float = 0.8, facecolor: str = 'r'
) -> Poly3DCollection:
    def get_sphere(inp):
        center = inp[:3]
        radius = inp[3]
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        return jnp.stack([x, y, z], axis=-1)

    get_sphere_vmap = jax.vmap(get_sphere)
    sphere_col = Poly3DCollection(
        merge01(get_sphere_vmap(jnp.concatenate([obstacles.center, obstacles.radius[:, None]], axis=-1))),
        alpha=alpha,
        linewidth=0.0,
        edgecolor='k',
        facecolor=facecolor
    )

    return sphere_col


def get_obs_collection(
        obstacles: Obstacle, color: str, alpha: float
):
    if isinstance(obstacles, Rectangle):
        n_obs = len(obstacles.center)
        obs_polys = [Polygon(obstacles.points[ii]) for ii in range(n_obs)]
        obs_col = PatchCollection(obs_polys, color="#8a0000", alpha=1.0, zorder=99)
    elif isinstance(obstacles, Cuboid):
        obs_col = get_cuboid_collection(obstacles, alpha=alpha, facecolor=color)
    elif isinstance(obstacles, Sphere):
        obs_col = get_sphere_collection(obstacles, alpha=alpha, facecolor=color)
    else:
        raise NotImplementedError
    return obs_col


def get_f1tenth_body(
        pos: Pos2d, theta: Array, delta: Array, radius: Array
):
    pos1 = pos + radius / 2 * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    pos2 = pos - radius / 2 * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    theta1 = theta + delta
    theta2 = theta
    pos = jnp.concatenate([pos1, pos2], axis=0)
    theta = jnp.concatenate([theta1, theta2], axis=0)
    body = jax.vmap(
        ft.partial(Rectangle.create, width=radius, height=radius / 4)
    )(pos, theta=theta)

    return body


def render_mpe(
        rollout: Rollout,
        video_path: pathlib.Path,
        side_length: float,
        dim: int,
        n_agent: int,
        n_obs: int,
        r: float,
        obs_r: float,
        cost_components: Tuple[str, ...],
        Ta_is_unsafe=None,
        viz_opts: dict = None,
        dpi: int = 100,
        n_goal: Optional[int] = None,
        **kwargs
):
    assert dim == 1 or dim == 2 or dim == 3
    if n_goal is None:
        n_goal = n_agent

    # set up visualization option
    if dim == 1 or dim == 2:
        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
    else:
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax: Axes3D = fig.add_subplot(projection='3d')
    ax.set_xlim(0., side_length)
    ax.set_ylim(0., side_length)
    if dim == 3:
        ax.set_zlim(0., side_length)
    ax.set(aspect="equal")
    if dim == 2:
        plt.axis("off")

    if viz_opts is None:
        viz_opts = {}

    # plot the first frame
    T_graph = rollout.graph
    graph0 = tree_index(T_graph, 0)

    agent_color = "#0068ff"
    goal_color = "#2fdd00"
    obs_color = "#8a0000"
    edge_goal_color = goal_color

    # plot obstacles
    if hasattr(graph0.env_states, "obs"):
        obs = graph0.env_states.obs
        if obs is not None:
            obs_circs = []
            for ii in range(len(obs)):
                obs_circs.append(plt.Circle(obs[ii, :2], obs_r, color=obs_color, linewidth=0.0))
            obs_col = MutablePatchCollection(obs_circs, match_original=True, zorder=1)
            ax.add_collection(obs_col)

    # plot agents
    n_color = [agent_color] * n_agent + [goal_color] * n_goal
    n_pos = np.array(graph0.states[:n_agent + n_goal, :dim]).astype(np.float32)
    n_radius = np.array([r] * (n_agent + n_goal))
    if dim == 1 or dim == 2:
        if dim == 1:
            n_pos = np.concatenate([n_pos, np.ones((n_agent + n_goal, 1)) * side_length / 2], axis=1)
        agent_circs = [plt.Circle(n_pos[ii], n_radius[ii], color=n_color[ii], linewidth=0.0)
                       for ii in range(n_agent + n_goal)]
        agent_col = MutablePatchCollection([i for i in reversed(agent_circs)], match_original=True, zorder=6)
        ax.add_collection(agent_col)
    else:
        plot_r = ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
        agent_col = ax.scatter(n_pos[:, 0], n_pos[:, 1], n_pos[:, 2],
                               s=plot_r, c=n_color, zorder=5)  # todo: the size of the agent might not be correct

    # plot edges
    all_pos = graph0.states[:n_agent + n_goal + n_obs, :dim]
    if dim == 1:
        all_pos = np.concatenate([all_pos, np.ones((n_agent + n_goal + n_obs, 1)) * side_length / 2], axis=1)
    edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
    is_pad = np.any(edge_index == n_agent + n_goal + n_obs, axis=0)
    e_edge_index = edge_index[:, ~is_pad]
    e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
    e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
    e_is_goal = (n_agent <= graph0.senders) & (graph0.senders < n_agent + n_goal)
    e_is_goal = e_is_goal[~is_pad]
    e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
    if dim == 1:
        e_lines = e_lines[~e_is_goal]
        e_colors = "0.2"
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    elif dim == 2:
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    else:
        edge_col = Line3DCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    ax.add_collection(edge_col)

    # text for cost and reward
    text_font_opts = dict(
        size=16,
        color="k",
        family="cursive",
        weight="normal",
        transform=ax.transAxes,
    )
    if dim == 1 or dim == 2:
        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
    else:
        cost_text = ax.text2D(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)

    # text for safety
    safe_text = []
    if Ta_is_unsafe is not None:
        if dim == 1 or dim == 2:
            safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
        else:
            safe_text = [ax.text2D(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]

    # text for time step
    if dim == 1 or dim == 2:
        kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
    else:
        kk_text = ax.text2D(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)

    # add agent labels
    label_font_opts = dict(
        size=20,
        color="k",
        family="cursive",
        weight="normal",
        ha="center",
        va="center",
        transform=ax.transData,
        clip_on=True,
        zorder=7,
    )
    agent_labels = []
    if dim == 1 or dim == 2:
        agent_labels = [ax.text(n_pos[ii, 0], n_pos[ii, 1], f"{ii}", **label_font_opts) for ii in range(n_agent)]
    else:
        for ii in range(n_agent):
            pos2d = proj3d.proj_transform(n_pos[ii, 0], n_pos[ii, 1], n_pos[ii, 2], ax.get_proj())[:2]
            agent_labels.append(ax.text2D(pos2d[0], pos2d[1], f"{ii}", **label_font_opts))

    # plot cbf
    cnt_col = []
    if "cbf" in viz_opts:
        if dim == 1 or dim == 3:
            print('Warning: CBF visualization is not supported in 1D or 3D.')
        else:
            Tb_xs, Tb_ys, Tbb_h, cbf_num = viz_opts["cbf"]
            bb_Xs, bb_Ys = np.meshgrid(Tb_xs[0], Tb_ys[0])
            norm = centered_norm(Tbb_h.min(), Tbb_h.max())
            levels = np.linspace(norm.vmin, norm.vmax, 15)

            cmap = get_BuRd().reversed()
            contour_opts = dict(cmap=cmap, norm=norm, levels=levels, alpha=0.9)
            cnt = ax.contourf(bb_Xs, bb_Ys, Tbb_h[0], **contour_opts)

            contour_line_opts = dict(levels=[0.0], colors=["k"], linewidths=3.0)
            cnt_line = ax.contour(bb_Xs, bb_Ys, Tbb_h[0], **contour_line_opts)

            cbar = fig.colorbar(cnt, ax=ax)
            cbar.add_lines(cnt_line)
            cbar.ax.tick_params(labelsize=36, labelfontfamily="Times New Roman")

            cnt_col = [*cnt.collections, *cnt_line.collections]

            ax.text(0.5, 1.0, "CBF for {}".format(cbf_num), transform=ax.transAxes, va="bottom")
    if "Vh" in viz_opts:
        if dim == 1 or dim == 2:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", zorder=100, **text_font_opts)
        else:
            Vh_text = ax.text2D(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

    # init function for animation
    def init_fn() -> list[plt.Artist]:
        return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, *cnt_col, kk_text]

    # update function for animation
    def update(kk: int) -> list[plt.Artist]:
        graph = tree_index(T_graph, kk)
        n_pos_t = graph.states[:-1, :dim]
        if dim == 1:
            n_pos_t = np.concatenate([n_pos_t, np.ones((n_agent + n_goal, 1)) * side_length / 2], axis=1)

        # update agent positions
        if dim == 1 or dim == 2:
            for ii in range(n_agent):
                agent_circs[ii].set_center(tuple(n_pos_t[ii]))
        else:
            agent_col.set_offsets(n_pos_t[:n_agent + n_goal, :2])
            agent_col.set_3d_properties(n_pos_t[:n_agent + n_goal, 2], zdir='z')

        # update edges
        e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
        is_pad_t = np.any(e_edge_index_t == n_agent + n_goal + n_obs, axis=0)
        e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
        e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
        e_is_goal_t = (n_agent <= graph.senders) & (graph.senders < n_agent + n_goal)
        e_is_goal_t = e_is_goal_t[~is_pad_t]
        e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
        e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
        if dim == 1:
            e_lines_t = e_lines_t[~e_is_goal_t]
            e_colors_t = "0.2"
        edge_col.set_segments(e_lines_t)
        edge_col.set_colors(e_colors_t)

        # update agent labels
        for ii in range(n_agent):
            if dim == 1 or dim == 2:
                agent_labels[ii].set_position(n_pos_t[ii])
            else:
                text_pos = proj3d.proj_transform(n_pos_t[ii, 0], n_pos_t[ii, 1], n_pos_t[ii, 2], ax.get_proj())[:2]
                agent_labels[ii].set_position(text_pos)

        # update cost and safe labels
        if kk < len(rollout.costs):
            all_costs = ""
            for i_cost in range(rollout.costs[kk].shape[1]):
                all_costs += f"    {cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
            all_costs = all_costs[:-2]
            cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
        else:
            cost_text.set_text("")
        if kk < len(Ta_is_unsafe):
            a_is_unsafe = Ta_is_unsafe[kk]
            unsafe_idx = np.where(a_is_unsafe)[0]
            safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
        else:
            safe_text[0].set_text("Unsafe: {}")

        # Update the contourf.
        nonlocal cnt, cnt_line
        if "cbf" in viz_opts and dim == 2:
            for c in cnt.collections:
                c.remove()
            for c in cnt_line.collections:
                c.remove()

            bb_Xs_t, bb_Ys_t = np.meshgrid(Tb_xs[kk], Tb_ys[kk])
            cnt = ax.contourf(bb_Xs_t, bb_Ys_t, Tbb_h[kk], **contour_opts)
            cnt_line = ax.contour(bb_Xs_t, bb_Ys_t, Tbb_h[kk], **contour_line_opts)

            cnt_col_t = [*cnt.collections, *cnt_line.collections]
        else:
            cnt_col_t = []

        if "Vh" in viz_opts:
            Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

        kk_text.set_text("kk={:04}".format(kk))

        return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, *cnt_col_t, kk_text]

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    anim_T = len(T_graph.n_node)
    ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
    save_anim(ani, video_path)


def render_lidar(
        rollout: Rollout,
        video_path: pathlib.Path,
        side_length: float,
        dim: int,
        n_agent: int,
        n_rays: int,
        r: float,
        cost_components: Tuple[str, ...],
        Ta_is_unsafe=None,
        viz_opts: dict = None,
        dpi: int = 100,
        n_goal: Optional[int] = None,
        **kwargs
):
    assert dim == 1 or dim == 2 or dim == 3
    if n_goal is None:
        n_goal = n_agent

    # set up visualization option
    if dim == 1 or dim == 2:
        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
    else:
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax: Axes3D = fig.add_subplot(projection='3d')
    ax.set_xlim(0., side_length)
    ax.set_ylim(0., side_length)
    if dim == 3:
        ax.set_zlim(0., side_length)
    ax.set(aspect="equal")
    if dim == 2:
        plt.axis("off")

    if viz_opts is None:
        viz_opts = {}

    # plot the first frame
    T_graph = rollout.graph
    graph0 = tree_index(T_graph, 0)

    agent_color = "#0068ff"
    goal_color = "#2fdd00"
    obs_color = "#8a0000"
    edge_goal_color = goal_color

    # plot obstacles
    if hasattr(graph0.env_states, "obstacle"):
        obs = graph0.env_states.obstacle
        if obs is not None:
            ax.add_collection(get_obs_collection(obs, obs_color, alpha=0.8))

    # plot agents
    n_hits = n_agent * n_rays
    n_color = [agent_color] * n_agent + [goal_color] * n_goal
    n_pos = np.array(graph0.states[:n_agent + n_goal, :dim]).astype(np.float32)
    n_radius = np.array([r] * (n_agent + n_goal))
    if dim == 1 or dim == 2:
        if dim == 1:
            n_pos = np.concatenate([n_pos, np.ones((n_agent + n_goal, 1)) * side_length / 2], axis=1)
        agent_circs = [plt.Circle(n_pos[ii], n_radius[ii], color=n_color[ii], linewidth=0.0)
                       for ii in range(n_agent + n_goal)]
        agent_col = MutablePatchCollection([i for i in reversed(agent_circs)], match_original=True, zorder=6)
        ax.add_collection(agent_col)
    else:
        plot_r = ax.transData.transform([r, 0])[0] - ax.transData.transform([0, 0])[0]
        agent_col = ax.scatter(n_pos[:, 0], n_pos[:, 1], n_pos[:, 2],
                               s=plot_r, c=n_color, zorder=5)  # todo: the size of the agent might not be correct

    # plot edges
    all_pos = graph0.states[:n_agent + n_goal + n_hits, :dim]
    if dim == 1:
        all_pos = np.concatenate([all_pos, np.ones((n_agent + n_goal + n_hits, 1)) * side_length / 2], axis=1)
    edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
    is_pad = np.any(edge_index == n_agent + n_goal + n_hits, axis=0)
    e_edge_index = edge_index[:, ~is_pad]
    e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
    e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
    e_is_goal = (n_agent <= graph0.senders) & (graph0.senders < n_agent + n_goal)
    e_is_goal = e_is_goal[~is_pad]
    e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
    if dim == 1:
        e_lines = e_lines[~e_is_goal]
        e_colors = "0.2"
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    elif dim == 2:
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    else:
        edge_col = Line3DCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
    ax.add_collection(edge_col)

    # text for cost and reward
    text_font_opts = dict(
        size=16,
        color="k",
        family="cursive",
        weight="normal",
        transform=ax.transAxes,
    )
    if dim == 1 or dim == 2:
        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
    else:
        cost_text = ax.text2D(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)

    # text for safety
    safe_text = []
    if Ta_is_unsafe is not None:
        if dim == 1 or dim == 2:
            safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
        else:
            safe_text = [ax.text2D(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]

    # text for time step
    if dim == 1 or dim == 2:
        kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
    else:
        kk_text = ax.text2D(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)

    # add agent labels
    label_font_opts = dict(
        size=20,
        color="k",
        family="cursive",
        weight="normal",
        ha="center",
        va="center",
        transform=ax.transData,
        clip_on=True,
        zorder=7,
    )
    agent_labels = []
    if dim == 1 or dim == 2:
        agent_labels = [ax.text(n_pos[ii, 0], n_pos[ii, 1], f"{ii}", **label_font_opts) for ii in range(n_agent)]
    else:
        for ii in range(n_agent):
            pos2d = proj3d.proj_transform(n_pos[ii, 0], n_pos[ii, 1], n_pos[ii, 2], ax.get_proj())[:2]
            agent_labels.append(ax.text2D(pos2d[0], pos2d[1], f"{ii}", **label_font_opts))

    # plot cbf
    cnt_col = []
    if "cbf" in viz_opts:
        if dim == 1 or dim == 3:
            print('Warning: CBF visualization is not supported in 1D or 3D.')
        else:
            Tb_xs, Tb_ys, Tbb_h, cbf_num = viz_opts["cbf"]
            bb_Xs, bb_Ys = np.meshgrid(Tb_xs[0], Tb_ys[0])
            norm = centered_norm(Tbb_h.min(), Tbb_h.max())
            levels = np.linspace(norm.vmin, norm.vmax, 15)

            cmap = get_BuRd().reversed()
            contour_opts = dict(cmap=cmap, norm=norm, levels=levels, alpha=0.9)
            cnt = ax.contourf(bb_Xs, bb_Ys, Tbb_h[0], **contour_opts)

            contour_line_opts = dict(levels=[0.0], colors=["k"], linewidths=3.0)
            cnt_line = ax.contour(bb_Xs, bb_Ys, Tbb_h[0], **contour_line_opts)

            cbar = fig.colorbar(cnt, ax=ax)
            cbar.add_lines(cnt_line)
            cbar.ax.tick_params(labelsize=36, labelfontfamily="Times New Roman")

            cnt_col = [*cnt.collections, *cnt_line.collections]

            ax.text(0.5, 1.0, "CBF for {}".format(cbf_num), transform=ax.transAxes, va="bottom")
    if "Vh" in viz_opts:
        if dim == 1 or dim == 2:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", zorder=100, **text_font_opts)
        else:
            Vh_text = ax.text2D(0.99, 0.99, "Vh: []", va="top", ha="right", **text_font_opts)

    # init function for animation
    def init_fn() -> list[plt.Artist]:
        return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, *cnt_col, kk_text]

    # update function for animation
    def update(kk: int) -> list[plt.Artist]:
        graph = tree_index(T_graph, kk)
        n_pos_t = graph.states[:-1, :dim]
        if dim == 1:
            n_pos_t = np.concatenate([n_pos_t, np.ones((n_agent + n_goal, 1)) * side_length / 2], axis=1)

        # update agent positions
        if dim == 1 or dim == 2:
            for ii in range(n_agent):
                agent_circs[ii].set_center(tuple(n_pos_t[ii]))
        else:
            agent_col.set_offsets(n_pos_t[:n_agent + n_goal, :2])
            agent_col.set_3d_properties(n_pos_t[:n_agent + n_goal, 2], zdir='z')

        # update edges
        e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
        is_pad_t = np.any(e_edge_index_t == n_agent + n_goal + n_hits, axis=0)
        e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
        e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
        e_is_goal_t = (n_agent <= graph.senders) & (graph.senders < n_agent + n_goal)
        e_is_goal_t = e_is_goal_t[~is_pad_t]
        e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
        e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
        if dim == 1:
            e_lines_t = e_lines_t[~e_is_goal_t]
            e_colors_t = "0.2"
        edge_col.set_segments(e_lines_t)
        edge_col.set_colors(e_colors_t)

        # update agent labels
        for ii in range(n_agent):
            if dim == 1 or dim == 2:
                agent_labels[ii].set_position(n_pos_t[ii])
            else:
                text_pos = proj3d.proj_transform(n_pos_t[ii, 0], n_pos_t[ii, 1], n_pos_t[ii, 2], ax.get_proj())[:2]
                agent_labels[ii].set_position(text_pos)

        # update cost and safe labels
        if kk < len(rollout.costs):
            all_costs = ""
            for i_cost in range(rollout.costs[kk].shape[1]):
                all_costs += f"    {cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
            all_costs = all_costs[:-2]
            cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
        else:
            cost_text.set_text("")
        if kk < len(Ta_is_unsafe):
            a_is_unsafe = Ta_is_unsafe[kk]
            unsafe_idx = np.where(a_is_unsafe)[0]
            safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
        else:
            safe_text[0].set_text("Unsafe: {}")

        # Update the contourf.
        nonlocal cnt, cnt_line
        if "cbf" in viz_opts and dim == 2:
            for c in cnt.collections:
                c.remove()
            for c in cnt_line.collections:
                c.remove()

            bb_Xs_t, bb_Ys_t = np.meshgrid(Tb_xs[kk], Tb_ys[kk])
            cnt = ax.contourf(bb_Xs_t, bb_Ys_t, Tbb_h[kk], **contour_opts)
            cnt_line = ax.contour(bb_Xs_t, bb_Ys_t, Tbb_h[kk], **contour_line_opts)

            cnt_col_t = [*cnt.collections, *cnt_line.collections]
        else:
            cnt_col_t = []

        if "Vh" in viz_opts:
            Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

        kk_text.set_text("kk={:04}".format(kk))

        return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, *cnt_col_t, kk_text]

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    anim_T = len(T_graph.n_node)
    ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
    save_anim(ani, video_path)


def get_heading_indicator(pos: Pos2d, theta: Array, radius: Array) -> Rectangle:
    pos_indicator = pos - radius / 2 * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    body = Rectangle.create(pos_indicator, theta=theta, width=radius, height=radius / 4)
    return body


def render_graph(
        side_length: float,
        dim: int,
        T: int,
        video_path: pathlib.Path,
        settings: dict,
        T_data: dict,
        dpi: int = 100,
        fps: float = 30.0,
        **kwargs
):
    # check setting
    assert dim in [2]  # only support 2D for now
    assert 'n_static_node' in settings.keys()
    assert 'n_moving_node' in settings.keys()
    n_static_node = settings['n_static_node']
    n_moving_node = settings['n_moving_node']
    # node information
    if n_static_node > 0:
        assert "static_node_r" in settings.keys()
        assert "static_node_color" in settings.keys()
        assert "static_node_labels" in settings.keys()
        assert "static_node_pos" in T_data.keys()
    if n_moving_node > 0:
        assert "moving_node_r" in settings.keys()
        assert "moving_node_color" in settings.keys()
        assert "moving_node_labels" in settings.keys()
        assert "T_moving_node_pos" in T_data.keys()
    # edge information
    if "T_edge_index" in T_data.keys():
        assert "T_edge_node_pos" in T_data.keys()
        assert "T_edge_colors" in T_data.keys()
    # heading information
    if "T_heading" in T_data.keys():
        assert "heading_color" in settings.keys()
    # steering information
    if "T_steering" in T_data.keys():
        assert "steering_color" in settings.keys()
    # other labels
    if "Ta_is_unsafe" not in T_data.keys():
        T_data["Ta_is_unsafe"] = None
    if "T_costs" not in T_data.keys():
        T_data["T_costs"] = None
        settings["cost_components"] = []
    else:
        assert "cost_components" in settings.keys()
    if "T_rewards" not in T_data.keys():
        T_data["T_rewards"] = None

    # set up visualization option
    ax: Axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
    ax.set_xlim(0., side_length)
    ax.set_ylim(0., side_length)
    ax.set(aspect="equal")
    plt.axis("off")

    # plot the first frame
    # static obstacles
    if "static_obstacles" in T_data.keys():
        static_obstacles = T_data["static_obstacles"]
        assert "obstacle_color" in settings.keys()
        if static_obstacles is not None:
            ax.add_collection(get_obs_collection(static_obstacles, settings["obstacle_color"], alpha=0.8))

    # static nodes
    if n_static_node > 0:
        static_nodes = [plt.Circle(T_data["static_node_pos"][ii], settings["static_node_r"][ii],
                                   color=settings["static_node_color"][ii], linewidth=0.0)
                        for ii in range(n_static_node)]
        static_nodes_col = MutablePatchCollection([i for i in static_nodes], match_original=True, zorder=6)
        ax.add_collection(static_nodes_col)
    else:
        static_nodes = []

    # moving nodes
    if n_moving_node > 0:
        moving_nodes = [plt.Circle(T_data["T_moving_node_pos"][0][ii], settings["moving_node_r"][ii],
                                   color=settings["moving_node_color"][ii], linewidth=0.0)
                        for ii in range(n_moving_node)]
        moving_nodes_col = MutablePatchCollection([i for i in moving_nodes], match_original=True, zorder=8)
        ax.add_collection(moving_nodes_col)

        # add heading
        if "T_heading" in T_data.keys():
            get_heading_indicator_ = jax.jit(jax.vmap(get_heading_indicator))
            valid = jnp.logical_not(jnp.isnan(T_data["T_heading"][0]))
            heading_indicator = get_heading_indicator_(
                T_data["T_moving_node_pos"][0][valid],
                T_data["T_heading"][0][valid],
                jnp.array(settings["moving_node_r"])[valid]
            )
            heading_poly = []
            for ii in range(valid.sum()):
                heading_poly.append(Polygon(heading_indicator.points[ii]))
            heading_col = MutablePatchCollection(heading_poly, color=settings["heading_color"], alpha=1.0, zorder=10)
            ax.add_collection(heading_col)

        # add steering
        if "T_steering" in T_data.keys():
            get_steering_indicator_ = jax.jit(jax.vmap(get_steering_indicator))
            valid = jnp.logical_not(jnp.isnan(T_data["T_steering"][0]))
            steering_indicator = get_steering_indicator_(
                T_data["T_moving_node_pos"][0][valid],
                T_data["T_steering"][0][valid],
                jnp.array(settings["moving_node_r"])[valid]
            )
            steering_poly = []
            for ii in range(valid.sum()):
                steering_poly.append(Polygon(steering_indicator.points[ii]))
            steering_col = MutablePatchCollection(steering_poly, color=settings["steering_color"], alpha=1.0, zorder=10)
            ax.add_collection(steering_col)
    else:
        moving_nodes = []

    # edges
    if "T_edge_index" in T_data.keys():
        edge_node_pos = T_data["T_edge_node_pos"][0]
        e_edge_index = T_data["T_edge_index"][0]
        e_start, e_end = edge_node_pos[e_edge_index[0, :]], edge_node_pos[e_edge_index[1, :]]
        e_lines = np.stack([e_start, e_end], axis=1)
        edge_col = LineCollection(e_lines, colors=T_data["T_edge_colors"][0], linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(edge_col)
    else:
        edge_col = []

    # text for cost and reward
    text_font_opts = dict(
        size=16,
        color="k",
        family="cursive",
        weight="normal",
        transform=ax.transAxes,
    )
    cost_text = [ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)]

    # text for safety
    safe_text = []
    if T_data["Ta_is_unsafe"] is not None:
        safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]

    # text for time step
    kk_text = [ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)]

    # add node labels
    label_font_opts = dict(
        size=20,
        color="k",
        family="cursive",
        weight="normal",
        ha="center",
        va="center",
        transform=ax.transData,
        clip_on=True,
        zorder=99,
    )
    static_node_labels = []
    for ii in range(n_static_node):
        if settings["static_node_labels"][ii] is not None:
            static_node_labels.append(ax.text(T_data["static_node_pos"][ii][0], T_data["static_node_pos"][ii][1],
                                              settings["static_node_labels"][ii], **label_font_opts))
    moving_node_labels = []
    for ii in range(n_moving_node):
        if settings["moving_node_labels"][ii] is not None:
            moving_node_labels.append(
                ax.text(T_data["T_moving_node_pos"][0][ii][0], T_data["T_moving_node_pos"][0][ii][1],
                        settings["moving_node_labels"][ii], **label_font_opts))

    # init function for animation
    def init_fn() -> List[plt.Artist]:
        plot_objects = [*static_node_labels, *moving_node_labels, *cost_text, *safe_text, *kk_text]
        if n_static_node > 0:
            plot_objects.append(static_nodes_col)
        if n_moving_node > 0:
            plot_objects.append(moving_nodes_col)
        if "T_edge_index" in T_data.keys():
            plot_objects.append(edge_col)
        return plot_objects

    # update function for animation
    def update(kk: int) -> List[plt.Artist]:
        # update moving node positions
        if n_moving_node > 0:
            for ii in range(n_moving_node):
                moving_nodes[ii].set_center(T_data["T_moving_node_pos"][kk][ii])

            # update heading
            if "T_heading" in T_data.keys():
                valid_t = jnp.logical_not(jnp.isnan(T_data["T_heading"][kk]))
                heading_indicator_t = get_heading_indicator_(
                    T_data["T_moving_node_pos"][kk][valid_t],
                    T_data["T_heading"][kk][valid_t],
                    jnp.array(settings["moving_node_r"])[valid_t]
                )
                for ii in range(valid_t.sum()):
                    heading_poly[ii].set_xy(heading_indicator_t.points[ii])

            # update steering
            if "T_steering" in T_data.keys():
                valid_t = jnp.logical_not(jnp.isnan(T_data["T_steering"][kk]))
                steering_indicator_t = get_steering_indicator_(
                    T_data["T_moving_node_pos"][kk][valid_t],
                    T_data["T_steering"][kk][valid_t],
                    jnp.array(settings["moving_node_r"])[valid_t]
                )
                for ii in range(valid_t.sum()):
                    steering_poly[ii].set_xy(steering_indicator_t.points[ii])

        # update edges
        if "T_edge_index" in T_data.keys():
            e_edge_index_t = T_data["T_edge_index"][kk]
            edge_node_pos_t = T_data["T_edge_node_pos"][kk]
            e_start_t, e_end_t = edge_node_pos_t[e_edge_index_t[0, :]], edge_node_pos_t[e_edge_index_t[1, :]]
            e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
            edge_col.set_segments(e_lines_t)
            edge_col.set_colors(T_data["T_edge_colors"][kk])

        # update node labels
        moving_node_label_id = 0
        for ii in range(n_moving_node):
            if settings["moving_node_labels"][ii] is not None:
                moving_node_labels[moving_node_label_id].set_position(T_data["T_moving_node_pos"][kk][ii])
                moving_node_label_id += 1

        # update cost and reward
        cost_text_ = ""
        if T_data["T_costs"] is not None:
            all_costs = ""
            for i_cost in range(T_data["T_costs"][kk].shape[1]):
                all_costs += (f"    {settings['cost_components'][i_cost]}: "
                              f"{T_data['T_costs'][kk][:, i_cost].max():5.4f}\n")
            all_costs = all_costs[:-2]
            cost_text_ += f"Cost:\n{all_costs}\n"
        if T_data["T_rewards"] is not None:
            cost_text_ += f"Reward: {T_data['T_rewards'][kk]:5.4f}"
        cost_text[0].set_text(cost_text_)

        # update safe labels
        if T_data["Ta_is_unsafe"] is not None:
            a_is_unsafe = T_data["Ta_is_unsafe"][kk]
            unsafe_idx = np.where(a_is_unsafe)[0]
            safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))

        # update time step
        kk_text[0].set_text("kk={:04}".format(kk))

        plot_objects = [*static_node_labels, *moving_node_labels, *cost_text, *safe_text, *kk_text]
        if n_static_node > 0:
            plot_objects.append(static_nodes_col)
        if n_moving_node > 0:
            plot_objects.append(moving_nodes_col)
        if "T_edge_index" in T_data.keys():
            plot_objects.append(edge_col)
        return plot_objects

    spf = 1 / fps
    mspf = 1_000 * spf
    ani = FuncAnimation(fig, update, frames=T, init_func=init_fn, interval=mspf, blit=True)
    save_anim(ani, video_path)
