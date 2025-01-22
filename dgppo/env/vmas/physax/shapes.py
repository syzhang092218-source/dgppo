import jax.numpy as jnp

from .jax_types import Pos2


class Shape:
    def moment_of_inertia(self, mass: float):
        raise NotImplementedError


class Box(Shape):
    def __init__(self, length: float = 0.3, width: float = 0.1, hollow: bool = False):
        super().__init__()
        assert length > 0, f"Length must be > 0, got {length}"
        assert width > 0, f"Width must be > 0, got {length}"
        self._length = length
        self._width = width
        self.hollow = hollow

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def get_delta_from_anchor(self, anchor: Pos2) -> Pos2:
        return anchor * jnp.array([self.length / 2, self.width / 2])
        # return anchor[X] * self.length / 2, anchor[Y] * self.width / 2

    def moment_of_inertia(self, mass: float):
        return (1 / 12) * mass * (self.length**2 + self.width**2)

    def circumscribed_radius(self):
        return jnp.sqrt((self.length / 2) ** 2 + (self.width / 2) ** 2)

    # def get_geometry(self) -> "Geom":
    #     from vmas.simulator import rendering
    #
    #     l, r, t, b = (
    #         -self.length / 2,
    #         self.length / 2,
    #         self.width / 2,
    #         -self.width / 2,
    #     )
    #     return rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])


class Sphere(Shape):
    def __init__(self, radius: float = 0.05):
        super().__init__()
        assert radius > 0, f"Radius must be > 0, got {radius}"
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    def get_delta_from_anchor(self, anchor: Pos2) -> Pos2:
        delta = self.radius * anchor
        delta_norm = jnp.linalg.vector_norm(delta)
        # If delta_norm is larger than the radius, scale it down to 1 / radius.
        delta_mult = jnp.where(delta_norm > self.radius, 1 / (self.radius * delta_norm), 1)
        delta = delta * delta_mult
        # if delta_norm > self.radius:
        #     delta /= delta_norm * self.radius
        return delta

    def moment_of_inertia(self, mass: float):
        return (1 / 2) * mass * self.radius**2

    def circumscribed_radius(self):
        return self.radius

    # def get_geometry(self) -> "Geom":
    #     from vmas.simulator import rendering
    #
    #     return rendering.make_circle(self.radius)


class Line(Shape):
    def __init__(self, length: float = 0.5):
        super().__init__()
        assert length > 0, f"Length must be > 0, got {length}"
        self._length = length
        self._width = 2

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def moment_of_inertia(self, mass: float):
        return (1 / 12) * mass * (self.length**2)

    def circumscribed_radius(self):
        return self.length / 2

    def get_delta_from_anchor(self, anchor: Pos2) -> Pos2:
        return anchor[0] * self.length / 2, 0.0

    # def get_geometry(self) -> "Geom":
    #     from vmas.simulator import rendering
    #
    #     return rendering.Line(
    #         (-self.length / 2, 0),
    #         (self.length / 2, 0),
    #         width=self.width,
    #     )
