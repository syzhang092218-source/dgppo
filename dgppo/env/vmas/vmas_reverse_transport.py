import pathlib
import einops as ei
import jax
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from typing import NamedTuple, Optional, Tuple
from matplotlib.animation import FuncAnimation

from .physax.entity import Agent, Entity
from .physax.shapes import Box, Sphere
from .physax.world import World
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from dgppo.utils.typing import Action, Array, Cost, Done, Info, Reward, State
from dgppo.utils.utils import save_anim, tree_index
from dgppo.env.base import MultiAgentEnv
from dgppo.env.utils import get_node_goal_rng


class VMASReverseTransportState(NamedTuple):
    box_pos: Array
    box_vel: Array
    a_pos: Array
    a_vel: Array
    goal_pos: Array
    o_pos: Array


class VMASReverseTransport(MultiAgentEnv):
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
            params: dict = None
    ):
        half_width = 0.8
        area_size = 2 * half_width
        self.half_width = half_width
        self.agent_radius = 0.03
        super().__init__(num_agents, area_size, max_step, dt, params)

        self.package_width = 0.6
        self.package_length = 0.6
        self.package_mass = 10.0

        self.obs_radius = 0.15
        self.n_obs = 3

        self.frame_skip = 4

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        # [pos(2), vel(2), box_pos(2), box_vel(2), rel_goal_pos(2), in_contact(1), rel_obs_pos_vec(6), rel_obs_dist(3)]
        return 20

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
        return "agent collisions", "obstacle collisions"

    def reset(self, key: Array) -> GraphsTuple:
        box_key, agent_key, a_vel_key, goal_key, obs_key = jax.random.split(key, 5)

        box_cen_halfwidth = self.half_width - 0.5 * self.package_length

        # Sample x0 position. Always near the edge.
        x0_radius = 0.98 * box_cen_halfwidth
        x0_angle = jax.random.uniform(box_key, minval=0.0, maxval=2 * np.pi)
        box_pos = x0_radius * jnp.array([jnp.cos(x0_angle), jnp.sin(x0_angle)])

        # Sample goal position. Always opposite the x0 position, but with some noise.
        goal_radius = x0_radius
        noise_ub = np.deg2rad(30)
        goal_angle = x0_angle + np.pi + jax.random.uniform(goal_key, minval=-noise_ub, maxval=noise_ub)
        goal_pos = goal_radius * jnp.array([jnp.cos(goal_angle), jnp.sin(goal_angle)])

        # Sample random obstacle locations inside radius so that by construction no collide with either start or goal.
        obs_radius = x0_radius - 1.5 * self.obs_radius
        assert obs_radius > 0

        o_angle = jax.random.uniform(obs_key, shape=(self.n_obs,), minval=0.0, maxval=2 * np.pi)
        o_pos = obs_radius * jnp.stack([jnp.cos(o_angle), jnp.sin(o_angle)], axis=-1)

        # Sample agent positions inside the box so they don't collide with each other.
        agent_pos, _ = get_node_goal_rng(
            agent_key,
            0.4 * self.package_length,
            2,
            self.num_agents,
            2 * self.params["agent_radius"],
            None
        )
        agent_pos = agent_pos - 0.2 + box_pos  # agents stay in the box

        box_vel = jnp.zeros(2)
        a_vel = jax.random.uniform(a_vel_key, shape=(self.num_agents, 2), minval=-0.01, maxval=0.01)

        env_state = VMASReverseTransportState(box_pos, box_vel, agent_pos, a_vel, goal_pos, o_pos)
        return self.get_graph(env_state)

    def step(
            self, graph: GraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[GraphsTuple, Reward, Cost, Done, Info]:
        action = self.clip_action(action)
        assert action.shape == (self.num_agents, 2)

        env_state: VMASReverseTransportState = graph.env_states

        world = World(x_semidim=1.2, y_semidim=1.2, contact_margin=6e-3, substeps=5, collision_force=500)

        def is_box(other: Entity):
            return other.name == "box"

        agent_names = [f"agent_{ii}" for ii in range(self.num_agents)]
        agents = [
            Agent.create(
                agent_names[ii],
                u_multiplier=0.5,
                shape=Sphere(self.agent_radius),
                collision_filter=is_box,
            )
            for ii in range(self.num_agents)
        ]
        box = Entity.create(
            "box",
            movable=True,
            rotatable=False,
            collide=True,
            shape=Box(length=self.package_length, width=self.package_width, hollow=True),
            mass=self.package_mass,
        )

        for ii, agent in enumerate(agents):
            agent = agent.withstate(pos=env_state.a_pos[ii], vel=env_state.a_vel[ii])
            agent = agent.withforce(force=action[ii] * agent.u_multiplier)
            agents[ii] = agent
        box = box.withstate(pos=env_state.box_pos, vel=env_state.box_vel)
        entities = [box, *agents]

        if self.frame_skip > 1:

            def body(entities_, _):
                entities_, _ = world.step(entities_)
                return entities_, None

            entities_secondlast, _ = lax.scan(body, entities, length=self.frame_skip - 1)
        else:
            entities_secondlast = entities

        entities, info = world.step(entities_secondlast)

        box = entities[0]
        agents = entities[1:]

        a_pos = jnp.stack([agent.state.pos for agent in agents], axis=0)
        a_vel = jnp.stack([agent.state.vel for agent in agents], axis=0)

        box_pos = box.state.pos
        box_vel = box.state.vel

        assert box_pos.shape == (2,)
        assert box_vel.shape == (2,)
        assert a_pos.shape == (self.num_agents, 2)
        assert a_vel.shape == (self.num_agents, 2)

        env_state_new = env_state._replace(box_pos=box_pos, box_vel=box_vel, a_pos=a_pos, a_vel=a_vel)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # calculate reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)

        next_graph = self.get_graph(env_state_new)
        return next_graph, reward, cost, done, info

    def get_reward(self, graph: GraphsTuple, action: Action) -> Reward:
        env_state: VMASReverseTransportState = graph.env_states

        box_pos = env_state.box_pos
        goal_pos = env_state.goal_pos

        # goal distance penalty
        dist2goal = jnp.linalg.norm(goal_pos - box_pos, axis=-1)
        reward = -dist2goal.mean() * 0.01

        # not reaching goal penalty
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001

        return reward

    def get_cost(self, graph: GraphsTuple) -> Cost:
        env_state: VMASReverseTransportState = graph.env_states
        agent_pos = env_state.a_pos

        # collision between agents
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        # (n_agent, )
        a_cost_agent: Array = self.params["agent_radius"] * 2 - min_dist

        # Box center colliding with obstacle.
        o_dist = jnp.linalg.norm(env_state.box_pos - env_state.o_pos, axis=-1)
        assert o_dist.shape == (self.n_obs,)
        min_dist = jnp.min(o_dist)
        cost_box = self.obs_radius - min_dist
        a_cost_box = ei.repeat(cost_box, " -> n", n=self.num_agents)

        cost = jnp.stack([4 * a_cost_agent, 2 * a_cost_box], axis=1)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)
        assert cost.shape == (self.num_agents, self.n_cost)

        return cost

    def get_a_incontact(self, a_pos: jnp.ndim, box_pos: jnp.ndim):
        # Center the positions so the box is at (0, 0).
        a_pos = a_pos - box_pos

        # The agent is a sphere with radius .03
        # The box is a square with size [-0.6, 0.6]
        # Check for contact by checking if we exceed the x / y limits, since the box is AABB.
        eps = 1e-2
        length = self.package_width - eps

        a_incontact = jnp.any(jnp.abs(a_pos) > length, axis=1)
        return a_incontact

    def get_graph(self, env_state: VMASReverseTransportState) -> GraphsTuple:
        state = env_state

        rel_goal_pos = state.goal_pos - state.box_pos
        a_incontact = self.get_a_incontact(state.a_pos, state.box_pos)

        o_rel_obspos = state.o_pos - state.box_pos
        assert o_rel_obspos.shape == (self.n_obs, 2)
        o_dist = jnp.sqrt(jnp.sum(o_rel_obspos ** 2, axis=-1) + 1e-6)
        o_rel_obspos_vec = o_rel_obspos / o_dist[:, None]

        idx_sort = jnp.argsort(o_dist)
        o_rel_obspos_vec = o_rel_obspos_vec[idx_sort]
        o_dist = o_dist[idx_sort]

        # node features.
        node_feats = jnp.zeros((self.num_agents, self.node_dim))
        node_feats = node_feats.at[:, :2].set(state.a_pos)
        node_feats = node_feats.at[:, 2:4].set(state.a_vel)
        node_feats = node_feats.at[:, 4:6].set(state.box_pos)
        node_feats = node_feats.at[:, 6:8].set(state.box_vel)
        node_feats = node_feats.at[:, 8:10].set(rel_goal_pos)
        node_feats = node_feats.at[:, 10].set(a_incontact)
        node_feats = node_feats.at[:, 11:17].set(o_rel_obspos_vec.flatten())
        node_feats = node_feats.at[:, 17:20].set(o_dist)

        node_type = jnp.full(self.num_agents, VMASReverseTransport.AGENT)
        edge_blocks = self.edge_blocks(env_state)

        # create graph
        n_state_vec = jnp.zeros((self.num_agents, 0))
        return GetGraph(node_feats, node_type, edge_blocks, env_state, n_state_vec).to_padded()

    def edge_blocks(self, env_state: VMASReverseTransportState) -> list[EdgeBlock]:
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
        T_env_states: VMASReverseTransportState = T_graph.env_states
        T_costs = rollout.costs

        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=dpi)
        ax.set_xlim(-1.01 * self.half_width, 1.01 * self.half_width)
        ax.set_ylim(-1.01 * self.half_width, 1.01 * self.half_width)
        ax.set_aspect("equal")

        # Plot a rectangle to visualize the halfwidth
        ax.add_patch(
            plt.Rectangle(
                (-self.half_width, -self.half_width), 2 * self.half_width, 2 * self.half_width, fc="none", ec="C3"
            )
        )

        # Plot a circle for the goal.
        goal_pos = T_env_states.goal_pos[0]
        dist2goal = self.params["dist2goal"]
        goal_circ = plt.Circle(goal_pos, dist2goal, color="C5", alpha=0.5)
        ax.add_patch(goal_circ)

        # Plot the obstacles.
        o_pos = T_env_states.o_pos[0]
        for oo in range(self.n_obs):
            obs_circ = plt.Circle(o_pos[oo], self.obs_radius, fc="C0", ec="none", alpha=0.7)
            ax.add_patch(obs_circ)

        # Plot the box.
        package_length, package_width = self.package_length, self.package_width
        center_offset = np.array([-package_length / 2, -package_width / 2])
        box_patch = plt.Rectangle(center_offset, package_length, package_width, ec="C3", fc="none")
        ax.add_patch(box_patch)

        # Plot the center of the box
        box_center = plt.Circle((0, 0), 0.5 * dist2goal, fc="C3", ec="none", zorder=6)
        ax.add_patch(box_center)

        # Plot agent
        # agent_colors = ["C2", "C1", "C4"]
        agent_colors = [f"C{i}" for i in range(self.num_agents)]
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
        cost_text = ax.text(0.99, 1.12, "cost=0", va="bottom", ha="right", **text_font_opts)

        # text for time step
        kk_text = ax.text(0.99, 1.08, "kk=0", va="bottom", ha="right", **text_font_opts)

        texts = [goal_text, obs_text, kk_text, cost_text]

        def init_fn() -> list[plt.Artist]:
            return [box_patch, box_center, *agent_patches, *texts]

        def update(kk: int) -> list[plt.Artist]:
            env_state: VMASReverseTransportState = tree_index(T_env_states, kk)

            # update agent positions
            for ii in range(self.num_agents):
                pos = env_state.a_pos[ii]
                assert pos.shape == (2,)
                agent_patches[ii].set_center(pos)

            # Update box position.
            box_patch.set_xy(center_offset + env_state.box_pos)
            box_center.set_center(env_state.box_pos)

            o_dist_obs = np.linalg.norm(env_state.box_pos - env_state.o_pos, axis=-1) - self.obs_radius
            dist_obs_str = ", ".join(["{:+.3f}".format(d) for d in o_dist_obs])
            dist_goal = np.linalg.norm(env_state.box_pos - env_state.goal_pos)

            cost_str = ", ".join(["{:+.3f}".format(c) for c in T_costs[kk].max(0)])

            obs_text.set_text("dist_obs=[{}]".format(dist_obs_str))
            goal_text.set_text("dist_goal={:.3f}".format(dist_goal))
            kk_text.set_text("kk={:04}".format(kk))
            cost_text.set_text("cost={}".format(cost_str))

            return [box_patch, box_center, *agent_patches, *texts]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        anim_T = len(T_graph.n_node)
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
        save_anim(ani, video_path)
