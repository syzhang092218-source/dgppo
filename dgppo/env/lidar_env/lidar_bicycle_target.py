import pathlib
import jax.numpy as jnp
import numpy as np
import jax.random as jr
import jax
import matplotlib.pyplot as plt

from typing import Tuple, Optional
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.pyplot import Axes

from dgppo.env.plot import get_obs_collection, get_f1tenth_body
from dgppo.env.utils import get_node_goal_rng
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import GraphsTuple
from dgppo.utils.typing import Action, Array, State, AgentState
from dgppo.env.lidar_env.base import LidarEnvState
from dgppo.env.lidar_env.lidar_target import LidarTarget
from dgppo.utils.utils import tree_index, MutablePatchCollection, save_anim


class LidarBicycleTarget(LidarTarget):

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = LidarBicycleTarget.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarBicycleTarget, self).__init__(num_agents, area_size, max_step, dt, params)

    @property
    def state_dim(self) -> int:
        return 5  # x, y, cos(theta), sin(theta), v

    @property
    def node_dim(self) -> int:
        return 8  # state dim (5) + indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def action_dim(self) -> int:
        return 2  # omega, acc

    def reset(self, key: Array) -> GraphsTuple:
        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (self._params["n_obs"], 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=-jnp.pi, maxval=jnp.pi)
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, self.num_agents, 2.2 * self.params["car_radius"], obstacles)
        theta_key, key = jr.split(key, 2)
        thetas = jr.uniform(theta_key, (self.num_agents,), minval=0, maxval=2 * np.pi)
        theta_states = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)
        states = jnp.concatenate([states, theta_states, jnp.zeros((self.num_agents, 1), dtype=states.dtype)], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 3), dtype=goals.dtype)], axis=1)
        env_states = LidarEnvState(states, goals, obstacles)

        # get lidar data
        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        def single_agent_step(x, u):
            theta = jnp.arctan2(x[3], x[2])
            theta_next = theta + x[4] * u[0] * self.dt * 10
            x_next = jnp.array([
                x[0] + x[4] * jnp.cos(theta) * self.dt,
                x[1] + x[4] * jnp.sin(theta) * self.dt,
                jnp.cos(theta_next),
                jnp.sin(theta_next),
                x[4] + u[1] * self.dt * 10.
            ])
            return x_next

        n_state_agent_new = jax.vmap(single_agent_step)(agent_states, action)

        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def state2feat(self, state: State) -> Array:
        vx = state[4] * state[2]
        vy = state[4] * state[3]
        feat = jnp.concatenate([state[:2], vx[None], vy[None]], axis=-1)
        assert feat.shape == (self.edge_dim,)
        return feat

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -1, -1, -0.5])
        upper_lim = jnp.array([self.area_size, self.area_size, 1, 1, 0.5])
        return lower_lim, upper_lim

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        n_rays = self.params["top_k_rays"] if self.params["n_obs"] > 0 else 0
        r = self.params["car_radius"]
        n_agent = self.num_agents
        n_hits = self.num_agents * n_rays
        n_goal = self.num_agents
        cost_components = self.cost_components

        # set up visualization option
        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
        ax.set_xlim(0., self.area_size)
        ax.set_ylim(0., self.area_size)
        ax.set(aspect="equal")
        plt.axis("off")

        # plot the first frame
        T_graph = rollout.graph
        T_action = rollout.actions
        graph0 = tree_index(T_graph, 0)

        agent_color = "#0068ff"
        goal_color = "#2fdd00"
        obs_color = "#8a0000"
        edge_goal_color = goal_color

        # plot obstacles
        if hasattr(graph0.env_states, "obstacle"):
            obs = graph0.env_states.obstacle
            ax.add_collection(get_obs_collection(obs, obs_color, alpha=0.8))

        # plot agents
        n_color = [agent_color] * n_agent + [goal_color] * n_goal
        n_pos = np.array(graph0.states[:n_agent + n_goal, :2]).astype(np.float32)
        n_radius = np.array([r] * (n_agent + n_goal))
        agent_circs = [plt.Circle(n_pos[ii], n_radius[ii], color=n_color[ii], linewidth=0.0)
                       for ii in range(n_agent + n_goal)]
        agent_col = MutablePatchCollection([i for i in reversed(agent_circs)], match_original=True, zorder=6)
        ax.add_collection(agent_col)

        # plot agent configurations
        agent_pos = graph0.states[:n_agent, :2]
        agent_theta = np.arctan2(graph0.states[:n_agent, 3], graph0.states[:n_agent, 2])
        agent_delta = T_action[0, :, 0]
        f1tenth_body = get_f1tenth_body(agent_pos, agent_theta, agent_delta, r)
        f1tenth_poly = [Polygon(f1tenth_body.points[ii]) for ii in range(len(f1tenth_body.center))]
        colors = ["#FFCC99"] * n_agent + ["#FF0000"] * n_agent
        f1tenth_col = MutablePatchCollection(f1tenth_poly, color=colors, alpha=1.0, zorder=99)
        ax.add_collection(f1tenth_col)

        # plot edges
        all_pos = graph0.states[:n_agent + n_goal + n_hits, :2]
        edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        is_pad = np.any(edge_index == n_agent + n_goal + n_hits, axis=0)
        e_edge_index = edge_index[:, ~is_pad]
        e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
        e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
        e_is_goal = (n_agent <= graph0.senders) & (graph0.senders < n_agent + n_goal)
        e_is_goal = e_is_goal[~is_pad]
        e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
        edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        ax.add_collection(edge_col)

        # text for cost and reward
        text_font_opts = dict(
            size=16,
            color="k",
            family="cursive",
            weight="normal",
            transform=ax.transAxes,
        )
        cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)

        # text for safety
        safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]

        # text for time step
        kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)

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
        agent_labels = [ax.text(n_pos[ii, 0], n_pos[ii, 1], f"{ii}", **label_font_opts) for ii in range(n_agent)]

        if "Vh" in viz_opts:
            Vh_text = ax.text(0.99, 0.99, "Vh: []", va="top", ha="right", zorder=100, **text_font_opts)

        # init function for animation
        def init_fn() -> list[plt.Artist]:
            return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, kk_text]

        # update function for animation
        def update(kk: int) -> list[plt.Artist]:
            graph = tree_index(T_graph, kk)
            n_pos_t = graph.states[:-1, :2]

            # update agent positions
            for ii in range(n_agent):
                agent_circs[ii].set_center(tuple(n_pos_t[ii]))

            # update f1tenth
            agent_theta_t = np.arctan2(graph.states[:n_agent, 3], graph.states[:n_agent, 2])
            agent_delta_t = T_action[kk, :, 0]
            agent_body_t = get_f1tenth_body(n_pos_t[:n_agent], agent_theta_t, agent_delta_t, r)
            for ii in range(n_agent * 2):
                f1tenth_poly[ii].set_xy(agent_body_t.points[ii])

            # update edges
            e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
            is_pad_t = np.any(e_edge_index_t == n_agent + n_goal + n_hits, axis=0)
            e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
            e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
            e_is_goal_t = (n_agent <= graph.senders) & (graph.senders < n_agent + n_goal)
            e_is_goal_t = e_is_goal_t[~is_pad_t]
            e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
            e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
            edge_col.set_segments(e_lines_t)
            edge_col.set_colors(e_colors_t)

            # update agent labels
            for ii in range(n_agent):
                agent_labels[ii].set_position(n_pos_t[ii])

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

            if "Vh" in viz_opts:
                Vh_text.set_text(f"Vh: {viz_opts['Vh'][kk]}")

            kk_text.set_text("kk={:04}".format(kk))

            return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, kk_text]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)
