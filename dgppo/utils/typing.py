from flax.core import FrozenDict
from jaxtyping import Array, Bool, Float, Int, Shaped
from typing import Dict, Any, Tuple
from numpy import ndarray


# jax types
PRNGKey = Float[Array, '2']

BoolScalar = Bool[Array, ""]
ABool = Bool[Array, "num_agents"]
Shape = Tuple[int, ...]

BFloat = Float[Array, "b"]
BInt = Int[Array, "b"]
FloatScalar = Float[Array, ""]
IntScalar = Int[Array, ""]
TFloat = Float[Array, "T"]

# environment types
Action = Float[Array, 'num_agents action_dim']
Reward = Float[Array, '']
Cost = Float[Array, 'nh']
Done = BoolScalar
Info = Dict[str, Shaped[Array, '']]
EdgeIndex = Float[Array, '2 n_edge']
AgentState = Float[Array, 'num_agents agent_state_dim']
State = Float[Array, 'num_states state_dim']
Node = Float[Array, 'num_nodes node_dim']
EdgeAttr = Float[Array, 'num_edges edge_dim']
Pos2d = Float[Array, '2']
Pos3d = Float[Array, '3']
Pos = Pos2d | Pos3d
Radius = Float[Array, '']


# neural network types
Params = Dict[str, Any]

# obstacles
ObsType = Int[Array, '']
ObsWidth = Float[Array, '']
ObsHeight = Float[Array, '']
ObsLength = Float[Array, '']
ObsTheta = Float[Array, '']
ObsQuaternion = Float[Array, '4']
