import pathlib
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import functools as ft

from typing import NamedTuple, Tuple, Optional
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes

from jaxtyping import Float
from matplotlib.collections import LineCollection

from ...trainer.data import Rollout
from ...utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ...utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from ...utils.utils import merge01, jax_vmap, tree_index, MutablePatchCollection, save_anim
from ..base import MultiAgentEnv
from dgppo.env.obstacle import Obstacle, Rectangle
from dgppo.env.plot import render_lidar, get_obs_collection, render_graph
from dgppo.env.utils import get_lidar, get_node_goal_rng
from dgppo.env.lidar_env.base import LidarEnv, LidarEnvState, LidarEnvGraphsTuple


class LidarDubinsCircle(LidarEnv):

    PARAMS = {
        "car_radius": 0.3,
        'R': 1.,
        "comm_radius": 5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "default_area_size": 3.0,
        "top_k_rays": 8,
        "v_tgt": 0.5
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = LidarDubinsCircle.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarDubinsCircle, self).__init__(num_agents, area_size, max_step, dt, params)
        assert self.num_agents == 1, "Only support single agent"

    @property
    def state_dim(self) -> int:
        return 5  # x, y, cos(theta), sin(theta), v

    @property
    def node_dim(self) -> int:
        return 8  # state dim (5) + indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        return 2  # omega, a

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions"

    def reset(self, key: Array) -> GraphsTuple:
        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        if n_rng_obs == 0:
            obstacles = None
        else:
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
            obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
            obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # agent starts randomly
        state_key, theta_key, key = jr.split(key, 3)
        states, _ = get_node_goal_rng(
            state_key, self.area_size, 2, self.num_agents, 2.2 * self.params["car_radius"], obstacles)
        theta = jr.uniform(theta_key, (self.num_agents,), minval=0, maxval=2 * np.pi)
        states = jnp.concatenate([states, jnp.cos(theta)[:, None], jnp.sin(theta)[:, None],
                                  jnp.zeros((self.num_agents, 1))], axis=-1)

        # states = jnp.array([[R + self.params["car_radius"], self.params["car_radius"], 1., 0., 0.]])

        # get a nominal goal
        goals = jnp.array([[self.area_size / 2, self.area_size / 2, 0., 0., 0.]])

        assert states.shape == (self.num_agents, self.state_dim)
        assert goals.shape == (self.num_goals, self.state_dim)
        env_states = LidarEnvState(states, goals, obstacles)

        # todo: add random moving obstacles

        # get lidar data
        lidar_data = self.get_lidar_data(states, obstacles)
        return self.get_graph(env_states, lidar_data)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """By default, use double integrator dynamics"""
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        agent_theta = jnp.arctan2(agent_states[:, 3], agent_states[:, 2])
        agent_theta_new = agent_theta + action[:, 0] * self.dt
        n_state_agent_new = jnp.concatenate([
            agent_states[:, 0] + agent_states[:, 4] * jnp.cos(agent_theta) * self.dt,
            agent_states[:, 1] + agent_states[:, 4] * jnp.sin(agent_theta) * self.dt,
            jnp.cos(agent_theta_new),
            jnp.sin(agent_theta_new),
            agent_states[:, 4] + action[:, 1] * self.dt * 5
        ], axis=-1)
        if n_state_agent_new.shape == (self.state_dim,):
            n_state_agent_new = n_state_agent_new[None, :]
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        reward = jnp.zeros(()).astype(jnp.float32)

        # agent should be on the circle

        center = jnp.array([self.area_size / 2, self.area_size / 2])
        agent_pos = agent_states[:, :2]
        dist2center = jnp.linalg.norm(agent_pos - center, axis=-1)
        reward -= ((dist2center - self.params["R"])**2).mean() * 0.01

        # velocity should be v_tgt
        v = agent_states[:, 4]
        reward -= ((v - self._params["v_tgt"])**2).mean() * 0.001

        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001

        return reward

    def state2feat(self, state: State) -> Array:
        # x_rel, y_rel, vx_rel, vy_rel
        return jnp.array([state[0], state[1], state[4] * state[2], state[4] * state[3]])

    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        edge_feats = (jax_vmap(self.state2feat)(state.agent)[:, None, :] -
                      jax_vmap(self.state2feat)(state.agent)[None, :, :])
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(edge_feats, agent_agent_mask, id_agent, id_agent)

        # agent - obs connection
        agent_obs_edges = []
        n_hits = self._params["top_k_rays"] * self.num_agents
        if lidar_data is not None:
            id_obs = jnp.arange(self.num_agents + self.num_goals, self.num_agents + self.num_goals + n_hits)
            for i in range(self.num_agents):
                id_hits = jnp.arange(i * self._params["top_k_rays"], (i + 1) * self._params["top_k_rays"])
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
                agent_obs_mask = jnp.ones((1, self._params["top_k_rays"]))
                agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
                lidar_feats = jnp.concatenate(
                    [lidar_feats, jnp.zeros((lidar_feats.shape[0], self.edge_dim - lidar_feats.shape[1]))], axis=-1)
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
                )

        return [agent_agent_edges] + agent_obs_edges

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -1, -1, -0.1])
        upper_lim = jnp.array([self.area_size, self.area_size, 1, 1, 0.5])
        return lower_lim, upper_lim

    def get_render_data(self, rollout: Rollout, Ta_is_unsafe: Array = None) -> Tuple[dict, dict]:
        n_hits = self._params["top_k_rays"] * self.num_agents if self.params["n_obs"] > 0 else 0
        graph0 = tree_index(rollout.graph, 0)
        n_moving_node = self.num_agents  # + self.num_goals + self.params['n_move_obs']

        T_moving_node_pos = []
        T_edge_index = []
        T_edge_node_pos = []
        T_edge_colors = []
        for kk in range(rollout.actions.shape[0]):
            graph_t = tree_index(rollout.graph, kk)

            # get positions of nodes
            agent_pos = graph_t.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
            # goal_pos = graph_t.type_states(type_idx=1, n_type=self.num_goals)[:, :2]
            # obs_pos = graph_t.type_states(type_idx=3, n_type=self.params['n_move_obs'])[:, :2]
            # T_moving_node_pos.append(jnp.concatenate([agent_pos], axis=0))
            T_moving_node_pos.append(agent_pos)

            # get edge index
            e_edge_index_t = np.stack([graph_t.senders, graph_t.receivers], axis=0)
            # is_pad_t = np.any(e_edge_index_t ==
            #                   self.num_agents + self.num_goals + n_hits + self.params['n_move_obs'], axis=0)
            is_pad_t = np.any(e_edge_index_t == self.num_agents + self.num_goals, axis=0)
            e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
            T_edge_index.append(e_edge_index_t)

            # get positions of nodes in edges
            T_edge_node_pos.append(graph_t.states[:, :2])

            # get edge colors
            # e_is_goal_t = (self.num_agents <= graph_t.senders) & (graph_t.senders < self.num_agents + self.num_goals)
            # e_is_goal_t = e_is_goal_t[~is_pad_t]
            # e_colors_t = ["#2fdd00" if e_is_goal_t[ii] else "0.2" for ii in range(e_edge_index_t.shape[1])]
            e_colors_t = ["0.2" for ii in range(e_edge_index_t.shape[1])]
            T_edge_colors.append(e_colors_t)

        settings = {
            "n_static_node": 0,
            "n_moving_node": n_moving_node,
            "moving_node_r": [self.params["car_radius"]] * n_moving_node,
            "moving_node_color": # ["#2fdd00"] * self.num_goals +
                                 # ["#ff0000"] * self.params['n_move_obs'] +
                                 ["#0068ff"] * self.num_agents,
            "moving_node_labels": # [None] * (self.num_goals + self.params['n_move_obs']) +
                                  [f"{i}" for i in range(self.num_agents)],
            "cost_components": self.cost_components,
            "obstacle_color": "#8a0000"
        }
        T_data = {
            "T_moving_node_pos": T_moving_node_pos,
            "Ta_is_unsafe": Ta_is_unsafe,
            "T_costs": rollout.costs,
            "T_rewards": rollout.rewards,
            "T_edge_index": T_edge_index,
            "T_edge_node_pos": T_edge_node_pos,
            "T_edge_colors": T_edge_colors,
            "static_obstacles": graph0.env_states.obstacle
        }
        # return settings, T_data

        # settings, T_data = super(LidarDubinsCircle, self).get_render_data(rollout, Ta_is_unsafe)

        # add heading and steering angle
        T_heading = []
        # T_steering = []
        for kk in range(rollout.actions.shape[0]):
            graph_t = tree_index(rollout.graph, kk)
            agent_states = graph_t.type_states(type_idx=0, n_type=self.num_agents)
            thetas = jnp.arctan2(agent_states[:, 3], agent_states[:, 2])
            # invalid_thetas = jnp.full((self.num_goals,), jnp.nan)
            # T_heading.append(jnp.concatenate([invalid_thetas, thetas]))
            T_heading.append(thetas)
            # T_steering.append(jnp.concatenate([invalid_thetas, rollout.actions[kk, :, 0] * jnp.deg2rad(20) + thetas]))
        T_data["T_heading"] = T_heading
        # T_data["T_steering"] = T_steering
        settings["heading_color"] = "#ffed47"
        # settings["steering_color"] = "#ffffff"

        return settings, T_data

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:

        settings, T_data = self.get_render_data(rollout, Ta_is_unsafe)

        render_graph(
            side_length=self.area_size,
            dim=2,
            T=rollout.actions.shape[0],
            video_path=video_path,
            settings=settings,
            T_data=T_data,
            dpi=dpi,
            **kwargs
        )

        # n_goal = 1
        #
        # # set up visualization option
        # side_length = self.area_size
        # ax: Axes
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
        # ax.set_xlim(0., side_length)
        # ax.set_ylim(0., side_length)
        # ax.set(aspect="equal")
        # plt.axis("on")
        #
        # if viz_opts is None:
        #     viz_opts = {}
        #
        # # plot the first frame
        # T_graph = rollout.graph
        # graph0 = tree_index(T_graph, 0)
        #
        # agent_color = "#0068ff"
        # goal_color = "#2fdd00"
        # obs_color = "#8a0000"
        # edge_goal_color = goal_color
        #
        # # plot obstacles
        # if hasattr(graph0.env_states, "obstacle"):
        #     obs = graph0.env_states.obstacle
        #     if obs is not None:
        #         ax.add_collection(get_obs_collection(obs, obs_color, alpha=0.8))
        #
        # # plot the tracking trajectory
        # tgt_traj = plt.Circle((side_length / 2, side_length / 2), self._params["R"],
        #                       color=goal_color, linewidth=1.0, fill=False)
        # ax.add_patch(tgt_traj)
        #
        # # plot agents
        # n_agent = self.num_agents
        # r = self._params["car_radius"]
        # n_rays = self._params["n_rays"] if self.params["n_obs"] > 0 else 0
        # n_hits = n_agent * n_rays
        # n_color = [agent_color] * n_agent
        # n_pos = np.array(graph0.states[:n_agent, :2]).astype(np.float32)
        # n_radius = np.array([r] * n_agent)
        # agent_circs = [plt.Circle(n_pos[ii], n_radius[ii], color=n_color[ii], linewidth=0.0)
        #                for ii in range(n_agent)]
        # agent_col = MutablePatchCollection([i for i in reversed(agent_circs)], match_original=True, zorder=6)
        # ax.add_collection(agent_col)
        #
        # # plot the heading
        # for ii in range(n_agent):
        #     agent_theta = np.arctan2(graph0.states[ii, 3], graph0.states[ii, 2])
        #     ax.arrow(n_pos[ii, 0], n_pos[ii, 1], 0.5 * np.cos(agent_theta), 0.5 * np.sin(agent_theta),
        #              head_width=0.1, head_length=0.1, fc='k', ec='k', zorder=7)
        #
        # # plot edges
        # all_pos = graph0.states[:n_agent + n_goal + n_hits, :2]
        # edge_index = np.stack([graph0.senders, graph0.receivers], axis=0)
        # is_pad = np.any(edge_index == n_agent + n_goal + n_hits, axis=0)
        # # is_pad = is_pad | np.any(edge_index == n_agent + n_goal - 1, axis=0)  # do not plot goal-agent edges
        # e_edge_index = edge_index[:, ~is_pad]
        # e_start, e_end = all_pos[e_edge_index[0, :]], all_pos[e_edge_index[1, :]]
        # e_lines = np.stack([e_start, e_end], axis=1)  # (e, n_pts, dim)
        # e_is_goal = (n_agent <= graph0.senders) & (graph0.senders < n_agent + n_goal)
        # e_is_goal = e_is_goal[~is_pad]
        # e_colors = [edge_goal_color if e_is_goal[ii] else "0.2" for ii in range(len(e_start))]
        # edge_col = LineCollection(e_lines, colors=e_colors, linewidths=2, alpha=0.5, zorder=3)
        # ax.add_collection(edge_col)
        #
        # # text for cost and reward
        # text_font_opts = dict(
        #     size=16,
        #     color="k",
        #     family="cursive",
        #     weight="normal",
        #     transform=ax.transAxes,
        # )
        # cost_text = ax.text(0.02, 1.00, "Cost: 1.0\nReward: 1.0", va="bottom", **text_font_opts)
        #
        # # text for safety
        # safe_text = []
        # if Ta_is_unsafe is not None:
        #     safe_text = [ax.text(0.99, 1.00, "Unsafe: {}", va="bottom", ha="right", **text_font_opts)]
        #
        #
        # # text for time step
        # kk_text = ax.text(0.99, 1.04, "kk=0", va="bottom", ha="right", **text_font_opts)
        #
        # # add agent labels
        # label_font_opts = dict(
        #     size=20,
        #     color="k",
        #     family="cursive",
        #     weight="normal",
        #     ha="center",
        #     va="center",
        #     transform=ax.transData,
        #     clip_on=True,
        #     zorder=7,
        # )
        # agent_labels = [ax.text(n_pos[ii, 0], n_pos[ii, 1], f"{ii}", **label_font_opts) for ii in range(n_agent)]
        #
        # # init function for animation
        # def init_fn() -> list[plt.Artist]:
        #     return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, kk_text]
        #
        # # update function for animation
        # def update(kk: int) -> list[plt.Artist]:
        #     graph = tree_index(T_graph, kk)
        #     n_pos_t = graph.states[:-1, :2]
        #
        #     # update agent positions
        #     for ii in range(n_agent):
        #         agent_circs[ii].set_center(tuple(n_pos_t[ii]))
        #
        #     # update edges
        #     e_edge_index_t = np.stack([graph.senders, graph.receivers], axis=0)
        #     is_pad_t = np.any(e_edge_index_t == n_agent + n_goal + n_hits, axis=0)
        #     e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
        #     e_start_t, e_end_t = n_pos_t[e_edge_index_t[0, :]], n_pos_t[e_edge_index_t[1, :]]
        #     e_is_goal_t = (n_agent <= graph.senders) & (graph.senders < n_agent + n_goal)
        #     e_is_goal_t = e_is_goal_t[~is_pad_t]
        #     e_colors_t = [edge_goal_color if e_is_goal_t[ii] else "0.2" for ii in range(len(e_start_t))]
        #     e_lines_t = np.stack([e_start_t, e_end_t], axis=1)
        #     edge_col.set_segments(e_lines_t)
        #     edge_col.set_colors(e_colors_t)
        #
        #     # update agent labels
        #     for ii in range(n_agent):
        #         agent_labels[ii].set_position(n_pos_t[ii])
        #
        #     # update cost and safe labels
        #     if kk < len(rollout.costs):
        #         all_costs = ""
        #         for i_cost in range(rollout.costs[kk].shape[1]):
        #             all_costs += f"    {self.cost_components[i_cost]}: {rollout.costs[kk][:, i_cost].max():5.4f}\n"
        #         all_costs = all_costs[:-2]
        #         cost_text.set_text(f"Cost:\n{all_costs}\nReward: {rollout.rewards[kk]:5.4f}")
        #     else:
        #         cost_text.set_text("")
        #     if kk < len(Ta_is_unsafe):
        #         a_is_unsafe = Ta_is_unsafe[kk]
        #         unsafe_idx = np.where(a_is_unsafe)[0]
        #         safe_text[0].set_text("Unsafe: {}".format(unsafe_idx))
        #     else:
        #         safe_text[0].set_text("Unsafe: {}")
        #
        #     kk_text.set_text("kk={:04}".format(kk))
        #
        #     return [agent_col, edge_col, *agent_labels, cost_text, *safe_text, kk_text]
        #
        # fps = 30.0
        # spf = 1 / fps
        # mspf = 1_000 * spf
        # anim_T = len(T_graph.n_node)
        # ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        # save_anim(ani, video_path)
