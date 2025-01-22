from typing import Callable, NamedTuple

import jax.numpy as jnp
import jax_dataclasses as jdc
from typing_extensions import Self

from .jax_types import Pos2, RealScalar, Vec1, Vec2
from .shapes import Shape


class EntityState(NamedTuple):
    pos: Pos2
    vel: Vec2
    rot: Vec1
    ang_vel: Vec1

    @staticmethod
    def zero():
        return EntityState(
            pos=jnp.zeros(2),
            vel=jnp.zeros(2),
            rot=jnp.zeros(1),
            ang_vel=jnp.zeros(1),
        )


class AgentState(NamedTuple):
    pos: Pos2
    vel: Vec2
    rot: Vec1
    ang_vel: Vec1
    # Force from action
    force: Vec2
    torque: Vec1

    @staticmethod
    def zero():
        return AgentState(
            pos=jnp.zeros(2),
            vel=jnp.zeros(2),
            rot=jnp.zeros(1),
            ang_vel=jnp.zeros(1),
            #
            force=jnp.zeros(2),
            torque=jnp.zeros(1),
        )


@jdc.pytree_dataclass
class Entity:
    state: EntityState

    name: jdc.Static[str]
    # entity can move / be pushed
    movable: jdc.Static[bool]
    # entity can rotate
    rotatable: jdc.Static[bool]
    # entity collides with others
    collide: jdc.Static[bool]

    # mass
    mass: jdc.Static[float]
    # max speed
    max_speed: jdc.Static[float | None]
    # max angular velocity
    max_angvel: jdc.Static[float | None]
    v_range: jdc.Static[float | None]

    # collision filter
    collision_filter: jdc.Static[Callable[[Self], bool]]
    # drag
    drag: jdc.Static[float | None]
    # friction
    linear_friction: jdc.Static[float | None]
    angular_friction: jdc.Static[float | None]
    gravity: jdc.Static[float | None]

    # shape
    shape: jdc.Static[Shape]

    @staticmethod
    def create(
        name: str,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        mass: float = 1.0,
        shape: Shape = None,
        v_range: float = None,
        max_speed: float = None,
        max_angvel: float = None,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity=None,
        collision_filter: Callable[[Self], bool] = lambda _: True,
    ):
        assert shape is not None, "Shape must be provided"

        return Entity(
            state=EntityState.zero(),
            name=name,
            movable=movable,
            rotatable=rotatable,
            collide=collide,
            mass=mass,
            shape=shape,
            v_range=v_range,
            max_speed=max_speed,
            max_angvel=max_angvel,
            drag=drag,
            linear_friction=linear_friction,
            angular_friction=angular_friction,
            gravity=gravity,
            collision_filter=collision_filter,
        )

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def withstate(
        self,
        pos: Pos2 | None = None,
        vel: Vec2 | None = None,
        rot: RealScalar | None = None,
        ang_vel: RealScalar | None = None,
    ):
        if pos is None:
            pos = self.state.pos
        if vel is None:
            vel = self.state.vel
        if rot is None:
            rot = self.state.rot
        if ang_vel is None:
            ang_vel = self.state.ang_vel
        assert pos.shape == (2,)
        assert vel.shape == (2,)
        assert rot.shape == (1,)
        assert ang_vel.shape == (1,)

        new_state = EntityState(pos=pos, vel=vel, rot=rot, ang_vel=ang_vel)
        return jdc.replace(self, state=new_state)

    @property
    def moment_of_inertia(self):
        return self.shape.moment_of_inertia(self.mass)

    def collides(self, entity: Self):
        if not self.collide:
            return False
        return self.collision_filter(entity)


@jdc.pytree_dataclass
class Agent(Entity):
    state: AgentState
    # force constraints
    f_range: jdc.Static[float | None]
    max_f: jdc.Static[float | None]
    # torque constraints
    t_range: jdc.Static[float | None]
    max_t: jdc.Static[float | None]
    #
    u_multiplier: jdc.Static[float]

    @staticmethod
    def create(
        name: str,
        shape: Shape = None,
        movable: bool = True,
        rotatable: bool = True,
        collide: bool = True,
        density: float = 25.0,
        mass: float = 1.0,
        f_range: float = None,
        max_f: float = None,
        t_range: float = None,
        max_t: float = None,
        v_range: float = None,
        max_speed: float = None,
        max_angvel: float = None,
        #
        # u_range: float | Sequence[float] = 1.0,
        u_multiplier: float = 1.0,
        #
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity: float = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
    ):
        return Agent(
            state=AgentState.zero(),
            name=name,
            movable=movable,
            rotatable=rotatable,
            collide=collide,
            mass=mass,
            shape=shape,
            v_range=v_range,
            max_speed=max_speed,
            max_angvel=max_angvel,
            drag=drag,
            linear_friction=linear_friction,
            angular_friction=angular_friction,
            gravity=gravity,
            collision_filter=collision_filter,
            #
            f_range=f_range,
            max_f=max_f,
            t_range=t_range,
            max_t=max_t,
            #
            u_multiplier=u_multiplier,
        )

    def withstate(
        self,
        *,
        pos: Pos2 | None = None,
        vel: Vec2 | None = None,
        rot: RealScalar | None = None,
        ang_vel: RealScalar | None = None,
    ) -> Self:
        if pos is None:
            pos = self.state.pos
        if vel is None:
            vel = self.state.vel
        if rot is None:
            rot = self.state.rot
        if ang_vel is None:
            ang_vel = self.state.ang_vel

        force = self.state.force
        torque = self.state.torque

        assert pos.shape == (2,)
        assert vel.shape == (2,)
        assert rot.shape == (1,)
        assert ang_vel.shape == (1,)

        assert force.shape == (2,)
        assert torque.shape == (1,)

        new_state = self.state._replace(pos=pos, vel=vel, rot=rot, ang_vel=ang_vel)
        return jdc.replace(self, state=new_state)

    def withforce(self, *, force: Vec2 | None = None, torque: Vec1 | None = None):
        if force is None:
            force = self.state.force
        if torque is None:
            torque = self.state.torque

        new_state = self.state._replace(force=force, torque=torque)
        return jdc.replace(self, state=new_state)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name
