import jax.numpy as jnp
import pathlib
import numpy as np
import jax.random as jr
import jax
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from matplotlib.patches import Polygon
from matplotlib.pyplot import Axes

from dgppo.env.plot import get_obs_collection, get_f1tenth_body, render_graph
from dgppo.env.utils import get_node_goal_rng
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import GraphsTuple
from dgppo.utils.typing import Action, Array, State, AgentState
from dgppo.env.lidar_env.base import LidarEnvState
from dgppo.env.lidar_env.lidar_target import LidarTarget
from dgppo.utils.utils import tree_index, MutablePatchCollection, save_anim


class LidarDubinsTarget(LidarTarget):

    PARAMS = {
        "car_radius": 0.3,
        "comm_radius": 3,
        "n_rays": 32,
        "obs_angle": jnp.deg2rad(90),
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "n_move_obs": 0,
        "default_area_size": 2.5,
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
        return 8  # state dim (5) + indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def action_dim(self) -> int:
        return 2  # omega, acc

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
            theta_next = theta + u[0] * self.dt
            x_next = jnp.array([
                x[0] + x[4] * jnp.cos(theta) * self.dt,
                x[1] + x[4] * jnp.sin(theta) * self.dt,
                jnp.cos(theta_next),
                jnp.sin(theta_next),
                x[4] + u[1] * self.dt * 5.
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
        lower_lim = jnp.array([0., 0., -1, -1, -0.1])
        upper_lim = jnp.array([self.area_size, self.area_size, 1, 1, 0.5])
        return lower_lim, upper_lim

    def get_render_data(self, rollout: Rollout, Ta_is_unsafe: Array = None) -> Tuple[dict, dict]:
        settings, T_data = super(LidarDubinsTarget, self).get_render_data(rollout, Ta_is_unsafe)

        # add heading and steering angle
        T_heading = []
        # T_steering = []
        for kk in range(rollout.actions.shape[0]):
            graph_t = tree_index(rollout.graph, kk)
            agent_states = graph_t.type_states(type_idx=0, n_type=self.num_agents)
            thetas = jnp.arctan2(agent_states[:, 3], agent_states[:, 2])
            # invalid_thetas = jnp.full((self.num_goals + self.params['n_move_obs'],), jnp.nan)
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
