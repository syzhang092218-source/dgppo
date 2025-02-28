import jax.numpy as jnp
import pathlib
import numpy as np
import jax.random as jr
import jax
import matplotlib.pyplot as plt

from typing import Optional, Tuple, NamedTuple, List
from matplotlib.patches import Polygon
from matplotlib.pyplot import Axes

from dgppo.env.obstacle import Obstacle
from dgppo.env.plot import get_obs_collection, get_f1tenth_body, render_graph
from dgppo.env.utils import get_node_goal_rng
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import GraphsTuple, GetGraph, EdgeBlock
from dgppo.utils.typing import Action, Array, State, AgentState, Cost, Pos2d, Reward, Done, Info
from dgppo.env.lidar_env.base import LidarEnvState
from dgppo.env.lidar_env.lidar_target import LidarTarget
from dgppo.utils.utils import tree_index, MutablePatchCollection, save_anim, merge01, jax_vmap


class LidarMoveObsEnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: Obstacle
    move_obs: State

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


LidarMoveObsGraphsTuple = GraphsTuple[State, LidarMoveObsEnvState]


class LidarDubinsTarget(LidarTarget):
    AGENT = 0
    GOAL = 1
    OBSTACLE = 2
    MOVE_OBS = 3

    PARAMS = {
        "car_radius": 0.3,
        "comm_radius": 5,
        "n_rays": 32,
        "obs_angle": jnp.deg2rad(360),
        "obs_len_range": [0.1, 0.3],
        "n_obs": 0,
        "n_move_obs": 1,
        'move_obs_vel': 0.3,
        'move_obs_radius': 0.25,
        "default_area_size": 4.0,
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
        area_size = LidarDubinsTarget.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarDubinsTarget, self).__init__(num_agents, area_size, max_step, dt, params)

    @property
    def state_dim(self) -> int:
        return 5  # x, y, cos(theta), sin(theta), v

    @property
    def node_dim(self) -> int:
        return 9  # state dim (5) + indicator: agent: 0001, goal: 0010, obstacle: 0100, move_obs: 1000

    @property
    def action_dim(self) -> int:
        return 2  # omega, acc

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        # return "agent collisions", "obs collisions", "move obs collisions"
        return "obs collisions", "move obs collisions"

    def reset_test(self) -> GraphsTuple:
        states = jnp.array([[0.5, 1.5, 1., 0., 0.]])
        goals = jnp.array([[3.5, 1.5, 0., 0., 0.]])
        move_obstacles = jnp.array([[1.3, 1.5, 0., 0., 0.]])
        env_state = LidarMoveObsEnvState(states, goals, None, move_obstacles)
        return self.get_graph(env_state)

    def reset(self, key: Array) -> GraphsTuple:
        # return self.reset_test()
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
            obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=-jnp.pi, maxval=jnp.pi)
            obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, self.num_agents, 2.2 * self.params["car_radius"], obstacles)
        agent_pos = states[:, :2]
        theta_key, key = jr.split(key, 2)
        thetas = jr.uniform(theta_key, (self.num_agents,), minval=0, maxval=2 * np.pi)
        theta_states = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)
        states = jnp.concatenate([states, theta_states, jnp.zeros((self.num_agents, 1), dtype=states.dtype)], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 3), dtype=goals.dtype)], axis=1)

        # generate moving obstacles
        if self.params['n_move_obs'] == 0:
            move_obstacles = None
        else:
            def get_obs_(inp):
                this_key, _, center_theta_ = inp
                theta_key_, this_key = jr.split(this_key, 2)
                # theta_ = jr.uniform(theta_key_, (), minval=0, maxval=2 * jnp.pi)
                # center_ = (self._params['R'] * 2 / 3 * jnp.array([jnp.cos(center_theta_), jnp.sin(center_theta_)])
                #            + self.area_size / 2)  # center of the trajectory of the moving obstacle
                # assert center_.shape == (2,)
                # pos_ = center_ + (self._params['R'] * 2 / 3) * jnp.array([jnp.cos(theta_), jnp.sin(theta_)])
                pos_ = jr.uniform(this_key, (2,), minval=0, maxval=self.area_size)
                assert pos_.shape == (2,)
                return this_key, pos_, center_theta_

            def non_valid_obs_(inp):
                _, pos_, _ = inp
                dist_min_agents = jnp.linalg.norm(agent_pos - pos_, axis=1).min()
                return dist_min_agents < self._params["car_radius"] + self._params["move_obs_radius"]

            def get_valid_obs_(carry, inp):
                this_key, center_theta_ = inp
                theta_key_, this_key = jr.split(this_key, 2)
                # theta_ = jr.uniform(theta_key_, (), minval=0, maxval=2 * jnp.pi)
                # # center_theta_ = center_thetas[0]
                # center_ = (self._params['R'] * 2 / 3 * jnp.array([jnp.cos(center_theta_), jnp.sin(center_theta_)])
                #            + self.area_size / 2)  # center of the trajectory of the moving obstacle
                # assert center_.shape == (2,)
                # pos_ = center_ + (self._params['R'] * 2 / 3) * jnp.array([jnp.cos(theta_), jnp.sin(theta_)])
                pos_ = jr.uniform(this_key, (2,), minval=0, maxval=self.area_size)
                assert pos_.shape == (2,)
                _, valid_obs_, _ = jax.lax.while_loop(
                    non_valid_obs_, get_obs_, (this_key, pos_, center_theta_))
                return carry, valid_obs_

            obs_key, key = jr.split(key, 2)
            obs_key = jr.split(obs_key, self.params['n_move_obs'])
            center_thetas = jnp.linspace(0, 2 * jnp.pi, self.params['n_move_obs'] + 1)[:-1]
            _, obs_pos = jax.lax.scan(get_valid_obs_, None, (obs_key, center_thetas))
            v_dir = jnp.stack([jnp.cos(center_thetas - jnp.pi / 2), jnp.sin(center_thetas - jnp.pi / 2)], axis=-1)
            obs_vel = jnp.ones((self.params['n_move_obs'], 1)) * self._params['move_obs_vel']
            # move_obstacles = jnp.concatenate([obs_pos, v_dir, obs_vel], axis=-1)
            move_obstacles = jnp.concatenate([obs_pos, jnp.zeros_like(v_dir), jnp.zeros_like(obs_vel)], axis=-1)

        env_states = LidarMoveObsEnvState(states, goals, obstacles, move_obstacles)
        # env_states = LidarEnvState(states, goals, obstacles)

        # get lidar data
        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        # convert action
        #

        def single_agent_step(x, u):
            theta = jnp.arctan2(x[3], x[2])
            theta_next = theta + u[0] * 0.4 * self.dt
            x_next = jnp.array([
                x[0] + x[4] * jnp.cos(theta) * self.dt,
                x[1] + x[4] * jnp.sin(theta) * self.dt,
                jnp.cos(theta_next),
                jnp.sin(theta_next),
                x[4] + u[1] * self.dt * 10.
            ])
            return x_next

        n_state_agent_new = jax.vmap(single_agent_step)(agent_states, action)

        # jax.debug.breakpoint()

        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def state2feat(self, state: State) -> Array:
        vx = state[4] * state[2]
        vy = state[4] * state[3]
        feat = jnp.concatenate([state[:2], vx[None], vy[None]], axis=-1)
        assert feat.shape == (self.edge_dim,)
        return feat

    def step(
            self, graph: LidarMoveObsGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[LidarMoveObsGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        obstacles = graph.env_states.obstacle if self.params['n_obs'] > 0 else None

        # calculate next states
        action = self.clip_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        lidar_data_next = self.get_lidar_data(next_agent_states, obstacles)
        info = {}

        # # calculate next goals
        # thetas = jnp.arctan2(goals[:, 1] - self.area_size / 2, goals[:, 0] - self.area_size / 2)
        # thetas_next = thetas + self._params['goal_vel'] * self.dt / self._params['R']
        # next_goal_pos = jnp.stack([self.area_size / 2 + self._params['R'] * jnp.cos(thetas_next),
        #                            self.area_size / 2 + self._params['R'] * jnp.sin(thetas_next)], axis=-1)
        # next_goal_vel = jnp.stack([-self._params['goal_vel'] * jnp.sin(thetas_next),
        #                            self._params['goal_vel'] * jnp.cos(thetas_next)], axis=-1)
        # next_goals = goals.at[:, :2].set(next_goal_pos).at[:, 2:].set(next_goal_vel)

        # calculate next moving obstacles
        move_obs = graph.env_states.move_obs
        # center_thetas = jnp.linspace(0, 2 * jnp.pi, self.params['n_move_obs'] + 1)[:-1]
        # center_pos = (self._params['R'] * 2 / 3 *
        #               jnp.stack([jnp.cos(center_thetas), jnp.sin(center_thetas)], axis=-1) + self.area_size / 2)
        # obs_thetas = jnp.arctan2(move_obs[:, 1] - center_pos[:, 1], move_obs[:, 0] - center_pos[:, 0])
        # obs_thetas_next = obs_thetas + self._params['move_obs_vel'] * self.dt / (self._params['R'] * 2 / 3)
        # next_obs_pos = center_pos + (self._params['R'] * 2 / 3) * jnp.stack([jnp.cos(obs_thetas_next),
        #                                                                      jnp.sin(obs_thetas_next)], axis=-1)
        # next_obs_vel = self._params['move_obs_vel'] * jnp.stack([-jnp.sin(obs_thetas_next),
        #                                                          jnp.cos(obs_thetas_next)], axis=-1)
        # next_move_obs = jnp.concatenate([next_obs_pos, next_obs_vel], axis=-1)

        next_state = LidarMoveObsEnvState(next_agent_states, goals, obstacles, move_obs)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        assert reward.shape == tuple()

        return self.get_graph(next_state, lidar_data_next), reward, cost, done, info

    def get_cost(self, graph: LidarMoveObsGraphsTuple) -> Cost:
        cost_0 = super(LidarDubinsTarget, self).get_cost(graph)
        cost_0 = cost_0[:, 1][:, None]

        # collision between agents and moving obstacles
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        move_obs = graph.type_states(type_idx=3, n_type=self.params['n_move_obs'])[:, :2]
        dist = jnp.linalg.norm(agent_pos[:, None, :] - move_obs[None, :, :], axis=-1)
        move_obs_cost: Array = self.params['car_radius'] + self.params['move_obs_radius'] - dist.min(axis=1)

        eps = 0.5
        cost_1 = jnp.where(move_obs_cost <= 0.0, move_obs_cost - eps, move_obs_cost + eps)
        cost_1 = jnp.clip(cost_1, -1.0, 1.0)

        cost = jnp.concatenate([cost_0, cost_1[:, None]], axis=-1)
        return cost

    def edge_blocks(self, state: LidarMoveObsEnvState, lidar_data: Optional[Pos2d] = None) -> List[EdgeBlock]:
        lidar_target_env_state = LidarEnvState(state.agent, state.goal, state.obstacle)
        edge_blocks = super(LidarDubinsTarget, self).edge_blocks(lidar_target_env_state, lidar_data)
        # return edge_blocks

        # agent - moving obstacle connection
        agent_pos = state.agent[:, :2]
        move_obs = state.move_obs[:, :2]
        pos_diff = agent_pos[:, None, :] - move_obs[None, :, :]
        edge_feats = (jax_vmap(self.state2feat)(state.agent)[:, None, :] -
                      jax_vmap(self.state2feat)(state.move_obs)[None, :, :])
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        agent_obs_mask = jnp.less(dist, self._params['comm_radius'])

        # calculate the angle between the agent heading and the agent-obs direction
        obs_dir = -pos_diff
        obs_dir = obs_dir / jnp.linalg.norm(obs_dir, axis=-1, keepdims=True)
        agent_heading = state.agent[:, 2:4]
        # agent_heading = agent_heading / jnp.linalg.norm(agent_heading, axis=-1, keepdims=True)
        cos_angle = jnp.sum(agent_heading[:, None, :] * obs_dir[None, :, :], axis=-1)[0]
        assert cos_angle.shape == (self.num_agents, self.params['n_move_obs'])
        agent_obs_mask = jnp.logical_and(agent_obs_mask, jnp.greater(cos_angle, jnp.cos(self._params['obs_angle'] / 2)))

        id_agent = jnp.arange(self.num_agents)
        n_hits = self._params["top_k_rays"] * self.num_agents if self.params["n_obs"] > 0 else 0
        id_obs = self.num_agents + self.num_goals + n_hits + jnp.arange(self.params['n_move_obs'])
        agent_obs_edges = EdgeBlock(edge_feats, agent_obs_mask, id_agent, id_obs)

        edge_blocks.append(agent_obs_edges)
        return edge_blocks

    def get_graph(self, state: LidarMoveObsEnvState, lidar_data: Pos2d = None) -> GraphsTuple:
        n_hits = self._params["top_k_rays"] * self.num_agents if self.params["n_obs"] > 0 else 0
        n_move = self.params['n_move_obs']
        n_nodes = self.num_agents + self.num_goals + n_hits + n_move

        if lidar_data is not None:
            lidar_data = merge01(lidar_data)

        # node features
        # states
        node_feats = jnp.zeros((self.num_agents + self.num_goals + n_hits + n_move, self.node_dim))
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(state.agent)
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, :self.state_dim].set(
            state.goal)
        if lidar_data is not None:
            node_feats = node_feats.at[self.num_agents + self.num_goals:
                                       self.num_agents + self.num_goals + n_hits, :2].set(lidar_data)
        if n_move > 0:
            node_feats = node_feats.at[-n_move:, :self.state_dim].set(state.move_obs)

        # indicators
        node_feats = node_feats.at[:self.num_agents, self.state_dim + 3].set(1.)  # agent
        node_feats = (
            node_feats.at[self.num_agents: self.num_agents + self.num_goals, self.state_dim + 2].set(1.))  # goal
        if n_hits > 0:
            node_feats = node_feats.at[self.num_agents + self.num_goals:
                                       self.num_agents + self.num_goals + n_hits, self.state_dim + 1].set(1.)
        if n_move > 0:
            node_feats = node_feats.at[-n_move:, self.state_dim].set(1.)

        # node type
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[: self.num_agents].set(LidarDubinsTarget.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(LidarDubinsTarget.GOAL)
        if n_hits > 0:
            node_type = node_type.at[self.num_agents + self.num_goals:
                                     self.num_agents + self.num_goals + n_hits].set(LidarDubinsTarget.OBS)
        if n_move > 0:
            node_type = node_type.at[-n_move:].set(LidarDubinsTarget.MOVE_OBS)

        # edge blocks
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        states = jnp.concatenate([state.agent, state.goal], axis=0)
        if lidar_data is not None:
            lidar_states = jnp.concatenate(
                [lidar_data, jnp.zeros((n_hits, self.state_dim - lidar_data.shape[1]))], axis=1)
            states = jnp.concatenate([states, lidar_states], axis=0)
        if n_move > 0:
            states = jnp.concatenate([states, state.move_obs], axis=0)
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=states
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -1, -1, 0.])
        upper_lim = jnp.array([self.area_size, self.area_size, 1, 1, 0.5])
        return lower_lim, upper_lim

    def get_render_data(self, rollout: Rollout, Ta_is_unsafe: Array = None) -> Tuple[dict, dict]:
        n_hits = self._params["top_k_rays"] * self.num_agents if self.params["n_obs"] > 0 else 0
        graph0 = tree_index(rollout.graph, 0)
        n_moving_node = self.num_agents + self.num_goals + self.params['n_move_obs']

        T_moving_node_pos = []
        T_edge_index = []
        T_edge_node_pos = []
        T_edge_colors = []
        for kk in range(rollout.actions.shape[0]):
            graph_t = tree_index(rollout.graph, kk)

            # get positions of nodes
            agent_pos = graph_t.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
            goal_pos = graph_t.type_states(type_idx=1, n_type=self.num_goals)[:, :2]
            obs_pos = graph_t.type_states(type_idx=3, n_type=self.params['n_move_obs'])[:, :2]
            T_moving_node_pos.append(jnp.concatenate([goal_pos, obs_pos, agent_pos], axis=0))

            # get edge index
            e_edge_index_t = np.stack([graph_t.senders, graph_t.receivers], axis=0)
            is_pad_t = np.any(e_edge_index_t ==
                              self.num_agents + self.num_goals + n_hits + self.params['n_move_obs'], axis=0)
            e_edge_index_t = e_edge_index_t[:, ~is_pad_t]
            T_edge_index.append(e_edge_index_t)

            # get positions of nodes in edges
            T_edge_node_pos.append(graph_t.states[:, :2])

            # get edge colors
            e_is_goal_t = (self.num_agents <= graph_t.senders) & (graph_t.senders < self.num_agents + self.num_goals)
            e_is_goal_t = e_is_goal_t[~is_pad_t]
            e_colors_t = ["#2fdd00" if e_is_goal_t[ii] else "0.2" for ii in range(e_edge_index_t.shape[1])]
            T_edge_colors.append(e_colors_t)

        settings = {
            "n_static_node": 0,
            "n_moving_node": n_moving_node,
            "moving_node_r": [self.params["car_radius"]] * self.num_goals +
                             [self.params["move_obs_radius"]] * self.params['n_move_obs'] +
                             [self.params["car_radius"]] * self.num_agents,
            "moving_node_color": ["#2fdd00"] * self.num_goals +
                                 ["#ff0000"] * self.params['n_move_obs'] + ["#0068ff"] * self.num_agents,
            "moving_node_labels": [None] * (self.num_goals + self.params['n_move_obs']) +
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

        # # add heading and steering angle
        T_heading = []
        # T_steering = []
        for kk in range(rollout.actions.shape[0]):
            graph_t = tree_index(rollout.graph, kk)
            agent_states = graph_t.type_states(type_idx=0, n_type=self.num_agents)
            thetas = jnp.arctan2(agent_states[:, 3], agent_states[:, 2])
            invalid_thetas = jnp.full((self.num_goals + self.params['n_move_obs'],), jnp.nan)
            T_heading.append(jnp.concatenate([invalid_thetas, thetas]))
            # T_heading.append(thetas)
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
