import pathlib
import einops as ei
import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from typing import NamedTuple, Optional, Tuple
from jaxtyping import PRNGKeyArray
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge

from .physax.entity import Agent, Entity
from .physax.shapes import Line, Sphere
from .physax.world import World
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from dgppo.utils.typing import Action, Array, Cost, Done, FloatScalar, Info, Reward, State
from dgppo.utils.utils import save_anim, tree_index
from dgppo.env.base import MultiAgentEnv
from dgppo.env.utils import get_node_goal_rng


class VMASWheelState(NamedTuple):
    line_angle: FloatScalar
    line_angvel: FloatScalar
    a_pos: Array
    a_vel: Array
    a_contact_force: Array
    goal_angle: FloatScalar
    avoid_angle: FloatScalar


class VMASWheel(MultiAgentEnv):
    AGENT = 0

    PARAMS = {
        "comm_radius": 0.4,
        "default_area_size": 0.8,
        "dist2goal": 0.01,
        "agent_radius": 0.03,
    }

    def __init__(
        self,
        num_agents: int,
        area_size: Optional[float] = None,
        max_step: int = 64,
        dt: float = 0.1,
        params: dict = None,
    ):
        half_width = 1.2
        assert num_agents == 3, "VMASWheel only supports 3 agents."
        area_size = 2 * half_width
        self.half_width = half_width
        self.agent_radius = 0.03
        super().__init__(3, area_size, max_step, dt, params)

        self.line_length = 2.0
        self.obs_halfwidth_rad = np.deg2rad(15)
        self.obs_init_pad_rad = np.deg2rad(1)

        self.frame_skip = 3

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        return 13  # [pos(2), vel(2), line sincos(2), line angvel(1), contact_force(2), goal sincos(2), obs sincos(2)]

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        return 2  # fx, fy

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return ("agent collisions",)

    def reset(self, key: Array) -> GraphsTuple:
        line_rot_key, line_angvel_key, agent_key, a_vel_key, a_goal_key, a_avoid_key = jax.random.split(key, 6)

        # randomize the line angle
        line_angle = jax.random.uniform(line_rot_key, minval=-np.pi, maxval=np.pi)
        line_angvel = jax.random.uniform(line_angvel_key, minval=-0.05, maxval=0.05)

        # randomize agent position
        # This is from [0, 2 * half_width]
        agent_pos, _ = get_node_goal_rng(
            agent_key,
            0.99 * self.area_size,
            2,
            self.num_agents,
            2 * self.params["agent_radius"],
            None
        )
        # Shift it to [-half_width, half_width]
        agent_pos = agent_pos - self.half_width

        a_vel = jax.random.uniform(a_vel_key, shape=(self.num_agents, 2), minval=-0.01, maxval=0.01)
        a_contactforce = jnp.zeros((self.num_agents, 2))

        # Sample random goal angle.
        goalangle = jax.random.uniform(a_goal_key, minval=-np.pi, maxval=np.pi)
        # Sample a random angle. Make sure that it is at least halfwidth_rad + goal_pad_rad away from the goal angle.
        # Take angle wrapping into account.
        avoid_angle = sample_valid_avoid_angle(
            a_avoid_key, line_angle, goalangle, self.obs_halfwidth_rad + self.obs_init_pad_rad, goal_maxdist=np.pi / 2
        )

        env_state = VMASWheelState(line_angle, line_angvel, agent_pos, a_vel, a_contactforce, goalangle, avoid_angle)
        return self.get_graph(env_state)

    def step(
        self, graph: GraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[GraphsTuple, Reward, Cost, Done, Info]:
        action = self.clip_action(action)
        assert action.shape == (self.num_agents, 2)

        env_state: VMASWheelState = graph.env_states

        world = World(x_semidim=1.2, y_semidim=1.2)

        line_length = self.line_length
        line_mass = 15.0

        def is_line(other: Entity):
            return other.name == "line"

        agent_drag = 0.25
        line_drag = 0.015
        max_angvel_line = 0.6

        agent_names = [f"agent_{ii}" for ii in range(self.num_agents)]
        agents = [
            Agent.create(
                agent_names[ii],
                u_multiplier=0.6,
                shape=Sphere(self.agent_radius),
                collision_filter=is_line,
                drag=agent_drag,
            )
            for ii in range(self.num_agents)
        ]
        line = Entity.create(
            "line",
            movable=False,
            rotatable=True,
            collide=True,
            shape=Line(length=line_length),
            mass=line_mass,
            drag=line_drag,
            max_angvel=max_angvel_line,
        )

        for ii, agent in enumerate(agents):
            agent = agent.withstate(pos=env_state.a_pos[ii], vel=env_state.a_vel[ii])
            agent = agent.withforce(force=action[ii] * agent.u_multiplier)
            agents[ii] = agent
        line = line.withstate(rot=env_state.line_angle[None], ang_vel=env_state.line_angvel[None])
        entities = [line, *agents]

        if self.frame_skip > 1:

            def body(entities_, _):
                entities_, _ = world.step(entities_)
                return entities_, None

            entities_secondlast, _ = lax.scan(body, entities, length=self.frame_skip - 1)
        else:
            entities_secondlast = entities

        entities, info = world.step(entities_secondlast)

        line = entities[0]
        agents = entities[1:]

        a_pos = jnp.stack([agent.state.pos for agent in agents], axis=0)
        a_vel = jnp.stack([agent.state.vel for agent in agents], axis=0)
        contact_forces_dict: dict[str, Array] = info["contact_forces"]

        line_rot = line.state.rot.squeeze(-1)
        line_angvel = line.state.ang_vel.squeeze(-1)

        a_contact_forces = jnp.stack([contact_forces_dict[agent.name] for agent in agents], axis=0)

        assert line_rot.shape == tuple()
        assert line_angvel.shape == tuple()
        assert a_pos.shape == (self.num_agents, 2)
        assert a_vel.shape == (self.num_agents, 2)
        assert a_contact_forces.shape == (self.num_agents, 2)

        env_state_new = env_state._replace(
            line_angle=line_rot, line_angvel=line_angvel, a_pos=a_pos, a_vel=a_vel, a_contact_force=a_contact_forces
        )
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # calculate reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)

        next_graph = self.get_graph(env_state_new)
        return next_graph, reward, cost, done, info

    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        env_state: VMASWheelState = graph.env_states

        # Reward for turning the wheel clockwise at the desired velocity.
        # Should be in [-pi, pi]
        ang_diff = angle_dist(env_state.line_angle, env_state.goal_angle)
        # [-pi, pi] -> [-0.1, 0.1] -> [0, 0.01]
        ang_diff_sq = (0.1 * ang_diff / jnp.pi) ** 2
        # [0, 0.01] -> [-0.01, 0] -> [-0.005, 0]
        reward = -ang_diff_sq * 0.5

        # not reaching goal penalty
        dist2goal_deg = 1
        reward: FloatScalar = reward - jnp.where(ang_diff > np.deg2rad(dist2goal_deg), 1.0, 0.0).mean() * 0.005

        return reward

    def get_cost(self, graph: GraphsTuple) -> Cost:
        env_state: VMASWheelState = graph.env_states
        agent_pos = env_state.a_pos

        # collision between agents
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        # (n_agent, )
        cost_agent: Array = self.params["agent_radius"] * 2 - min_dist

        # Line colliding with the obstacle.
        line_dist = angle_dist(env_state.line_angle, env_state.avoid_angle)
        cost_line = self.obs_halfwidth_rad - jnp.abs(line_dist)
        cost_line = cost_line / np.pi
        # (n_agent,)
        a_cost_line = ei.repeat(cost_line, "-> a", a=self.num_agents)
        cost = jnp.stack([cost_agent, a_cost_line], axis=-1)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0)
        assert cost.shape == (self.num_agents, self.n_cost)

        return cost

    def get_graph(self, env_state: VMASWheelState) -> GraphsTuple:
        state = env_state

        sin, cos = jnp.sin(state.line_angle), jnp.cos(state.line_angle)
        sincos = jnp.array([sin, cos])

        ang_diff_goal = angle_dist(state.line_angle, state.goal_angle)
        sin, cos = jnp.sin(ang_diff_goal), jnp.cos(ang_diff_goal)
        sincos_goal = jnp.array([sin, cos])

        ang_diff_obs = angle_dist(state.line_angle, state.avoid_angle)
        sin, cos = jnp.sin(ang_diff_obs), jnp.cos(ang_diff_obs)
        sincos_obs = jnp.array([sin, cos])

        # node features.
        node_feats = jnp.zeros((self.num_agents, self.node_dim))
        node_feats = node_feats.at[:, :2].set(state.a_pos)
        node_feats = node_feats.at[:, 2:4].set(state.a_vel)
        node_feats = node_feats.at[:, 4:6].set(sincos)
        node_feats = node_feats.at[:, 6].set(state.line_angvel)
        node_feats = node_feats.at[:, 7:9].set(state.a_contact_force)
        node_feats = node_feats.at[:, 9:11].set(sincos_goal)
        node_feats = node_feats.at[:, 11:13].set(sincos_obs)

        node_type = jnp.full(self.num_agents, VMASWheel.AGENT)
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        n_state_vec = jnp.zeros((self.num_agents, 0))
        return GetGraph(node_feats, node_type, edge_blocks, env_state, n_state_vec).to_padded()

    def edge_blocks(self, env_state: VMASWheelState) -> list[EdgeBlock]:
        state = env_state

        nagent = self.num_agents
        agent_pos = state.a_pos
        agent_vel = state.a_vel
        agent_states = jnp.concatenate([agent_pos[:, :2], agent_vel[:, :2]], axis=-1)

        # agent - agent connection
        state_diff = agent_states[:, None, :] - agent_states[None, :, :]
        agent_agent_mask = jnp.array(jnp.eye(nagent) == 0)
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        return [agent_agent_edges]

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        pass

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    def render_video(
        self,
        rollout: Rollout,
        video_path: pathlib.Path,
        Ta_is_unsafe=None,
        viz_opts: dict = None,
        n_goal: int = None,
        dpi: int = 200,
        **kwargs,
    ) -> None:
        T_graph = rollout.graph
        T_env_states: VMASWheelState = T_graph.env_states

        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
        ax.set_xlim(-1.02 * self.half_width, 1.02 * self.half_width)
        ax.set_ylim(-1.02 * self.half_width, 1.02 * self.half_width)
        ax.set_aspect("equal")

        line_length = self.line_length
        width = 0.05

        # Plot a line for the goal.
        goal_angle = T_env_states.goal_angle[0]
        ax.plot([0, np.cos(goal_angle) * line_length], [0, np.sin(goal_angle) * line_length], "C5", lw=2, alpha=0.2)

        # Plot a sector for the obstacle.
        obs_angle = T_env_states.avoid_angle[0]
        obs_halfwidth_rad = self.obs_halfwidth_rad
        wedge = Wedge(
            (0, 0),
            1.2 * line_length,
            np.rad2deg(obs_angle - obs_halfwidth_rad),
            np.rad2deg(obs_angle + obs_halfwidth_rad),
            alpha=0.2,
            color="C0",
        )
        ax.add_patch(wedge)

        # Plot the line.
        # x: -length/2, length/2. y: -width/2, width/2.
        line_patch_pos = plt.Rectangle([0.0, -width / 2], line_length / 2, width, rotation_point=(0, 0), fc="C5")
        ax.add_patch(line_patch_pos)
        line_patch_neg = plt.Rectangle(
            [-line_length / 2, -width / 2], line_length / 2, width, rotation_point=(0, 0), fc="C3"
        )
        ax.add_patch(line_patch_neg)

        # Plot agent
        agent_colors = ["C2", "C1", "C4"]
        agent_radius = self.agent_radius
        agent_patches = [
            plt.Circle((0, 0), agent_radius, color=agent_colors[ii], zorder=5) for ii in range(self.num_agents)
        ]
        [ax.add_patch(patch) for patch in agent_patches]

        text_font_opts = dict(
            size=16,
            color="k",
            family="cursive",
            weight="normal",
            transform=ax.transAxes,
        )

        # text for line velocity
        goal_text = ax.text(0.99, 1.00, "dist_goal=0", va="bottom", ha="right", **text_font_opts)
        obs_text = ax.text(0.99, 1.04, "dist_obs=0", va="bottom", ha="right", **text_font_opts)
        angvel_text = ax.text(0.99, 1.12, r"$\omega$=0", va="bottom", ha="right", **text_font_opts)

        # text for time step
        kk_text = ax.text(0.99, 1.08, "kk=0", va="bottom", ha="right", **text_font_opts)

        texts = [goal_text, obs_text, angvel_text, kk_text]

        def init_fn() -> list[plt.Artist]:
            return [line_patch_pos, line_patch_neg, *agent_patches, *texts]

        def update(kk: int) -> list[plt.Artist]:
            env_state: VMASWheelState = tree_index(T_env_states, kk)

            # update agent positions
            for ii in range(self.num_agents):
                pos = env_state.a_pos[ii]
                assert pos.shape == (2,)
                agent_patches[ii].set_center(pos)

            # Update line rotation.
            line_patch_pos.angle = np.rad2deg(env_state.line_angle)
            line_patch_neg.angle = np.rad2deg(env_state.line_angle)

            dist_obs = angle_dist_np(env_state.line_angle, env_state.avoid_angle)
            dist_goal = angle_dist_np(env_state.line_angle, env_state.goal_angle)

            obs_text.set_text("dist_obs={:.3f}".format(dist_obs))
            goal_text.set_text("dist_goal={:.3f}".format(dist_goal))
            kk_text.set_text("kk={:04}".format(kk))
            angvel_text.set_text(r"$\omega$={:+.3f}".format(env_state.line_angvel))

            return [line_patch_pos, line_patch_neg, *agent_patches, *texts]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)


def angle_dist(a: FloatScalar, b: FloatScalar) -> FloatScalar:
    """Compute the shortest distance between two angles. Answer should be in [-pi, pi]."""
    return jnp.arctan2(jnp.sin(a - b), jnp.cos(a - b))


def angle_dist_np(a: float, b: float) -> float:
    """Compute the shortest distance between two angles. Answer should be in [-pi, pi]."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def sample_valid_avoid_angle(
        key: PRNGKeyArray,
        line_angle: FloatScalar,
        goal_angle: FloatScalar,
        min_dist: FloatScalar,
        goal_maxdist: FloatScalar
) -> FloatScalar:
    # Rejection sampling: Sample a bunch of random angles, pick the first one that satisfies the conditions.
    n = 8
    b_angles = jax.random.uniform(key, shape=(n,), minval=-np.pi, maxval=np.pi)
    b_disttogoal = jnp.abs(angle_dist(b_angles, goal_angle))
    b_disttoline = jnp.abs(angle_dist(b_angles, line_angle))
    b_valid = (b_disttogoal > min_dist) & (b_disttoline > min_dist) & (b_disttogoal < goal_maxdist)

    b_disttogoal_masked = jnp.where(b_valid, b_disttogoal, jnp.inf)
    b_argsort = jnp.argsort(b_disttogoal_masked)
    idx = b_argsort[0]
    return b_angles[idx]
