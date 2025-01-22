import jax.numpy as jnp
import jax.random as jr
import jax

from typing import Optional

from dgppo.env.utils import get_node_goal_rng, inside_obstacles
from dgppo.utils.graph import GraphsTuple
from dgppo.utils.typing import Action, Array, Pos2d, Reward
from dgppo.env.lidar_env.base import LidarEnvState, LidarEnvGraphsTuple
from dgppo.env.lidar_env.lidar_spread import LidarSpread


class LidarLine(LidarSpread):

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
        area_size = LidarLine.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarLine, self).__init__(num_agents, area_size, max_step, dt, params)
        self.num_goals = 2

    def reset(self, key: Array) -> GraphsTuple:
        # randomly generate agent
        states, _ = get_node_goal_rng(
            key,
            self.area_size,
            2,
            self.num_agents,
            2 * self.params["car_radius"],
            None,
        )

        # generate two landmarks
        min_dist = (self.num_agents - 2) * 6 * self.params["car_radius"]
        landmark0_key, key = jr.split(key)
        side = self.area_size - min_dist
        if side < 0:
            raise ValueError("The area size is too small to place the landmarks.")
        candidate = jr.uniform(landmark0_key, (2,),
                               minval=jnp.array([0, 0]),
                               maxval=jnp.array([self.area_size - side, side]))
        candidate = candidate - jnp.array([self.area_size / 2, 0]) + jnp.array([0, self.area_size / 2 - side])
        region_key, key = jr.split(key)
        region = jr.randint(region_key, (), minval=0, maxval=4)  # region id
        rotation_angle = region * jnp.pi / 2
        rotation_matrix = jnp.array([[jnp.cos(rotation_angle), -jnp.sin(rotation_angle)],
                                     [jnp.sin(rotation_angle), jnp.cos(rotation_angle)]])
        candidate = rotation_matrix @ candidate[:, None][:, 0]
        landmark0 = candidate + jnp.array([self.area_size / 2, self.area_size / 2])

        def get_landmark1(inp):
            this_key, _ = inp
            use_key, this_key = jr.split(this_key, 2)
            return this_key, jr.uniform(use_key, (2,), minval=0, maxval=self.area_size)

        def non_valid_landmark1(inp):
            _, this_goal = inp
            return jnp.linalg.norm(this_goal - landmark0) < min_dist

        landmark1_key, key = jr.split(key)
        landmark1_candidate = jr.uniform(landmark1_key, (2,), minval=0, maxval=self.area_size)
        _, landmark1 = jax.lax.while_loop(non_valid_landmark1, get_landmark1, (key, landmark1_candidate))
        landmarks = jnp.stack([landmark0, landmark1])
        goals = self.landmark2goal(landmarks)

        def get_obs(inp):
            this_key, _, _, _ = inp
            pos_key, length_key, theta_key, this_key = jr.split(this_key, 4)
            pos = jr.uniform(pos_key, (1, 2), minval=0, maxval=self.area_size)
            length = jr.uniform(
                length_key,
                (1, 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta = jr.uniform(theta_key, (1,), minval=0, maxval=jnp.pi)
            return this_key, pos, length, theta

        def non_valid_obs(inp):
            _, pos, length, theta = inp
            obs = self.create_obstacles(pos, length[:, 0], length[:, 1], theta)
            points = jnp.concatenate([states, goals], axis=0)
            return inside_obstacles(points, obs, r=self._params["car_radius"] * 1.1).max()

        def get_valid_obs(carry, inp):
            this_key = inp
            pos_key, length_key, theta_key, this_key = jr.split(this_key, 4)
            pos = jr.uniform(pos_key, (1, 2), minval=0, maxval=self.area_size)
            length = jr.uniform(
                length_key,
                (1, 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta = jr.uniform(theta_key, (1,), minval=0, maxval=jnp.pi)
            _, pos, length, theta = jax.lax.while_loop(non_valid_obs, get_obs, (this_key, pos, length, theta))
            return carry, (pos, length, theta)

        obs_key, key = jr.split(key)
        obs_keys = jr.split(obs_key, self.params["n_obs"])
        _, (obs_pos, obs_length, obs_theta) = jax.lax.scan(get_valid_obs, None, obs_keys)
        obstacles = self.create_obstacles(
            obs_pos.squeeze(1), obs_length[:, :, 0].squeeze(1), obs_length[:, :, 1].squeeze(1), obs_theta.squeeze(1))

        states = jnp.concatenate([states, jnp.zeros_like(states)], axis=1)
        landmarks = jnp.concatenate([landmarks, jnp.zeros_like(landmarks)], axis=1)
        env_states = LidarEnvState(states, landmarks, obstacles)

        # get lidar data
        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    def landmark2goal(self, landmarks: Pos2d) -> Pos2d:
        assert landmarks.shape == (2, 2)
        direction = landmarks[1] - landmarks[0]
        n_interval = self.num_agents - 1
        goals = landmarks[0] + jnp.arange(0, n_interval + 1)[:, None] * direction / n_interval
        return goals

    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        landmarks = graph.type_states(type_idx=1, n_type=2)[:, :2]
        goals = self.landmark2goal(landmarks)
        reward = jnp.zeros(()).astype(jnp.float32)

        # each goal finds the nearest agent
        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(jnp.expand_dims(goal_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1).min(axis=1)
        reward -= dist2goal.mean() * 0.01

        # not reaching goal penalty
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001

        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001

        return reward
