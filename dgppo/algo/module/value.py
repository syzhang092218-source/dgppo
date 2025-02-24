import functools as ft
import flax.linen as nn
import jax.numpy as jnp

from typing import Type, Any, Tuple

from ...nn.mlp import MLP
from ...nn.gnn import GraphTransformerGNN, GNN
from ...nn.rnn import RNN
from ...nn.utils import default_nn_init
from ...utils.typing import Array, Params, PRNGKey
from ...utils.graph import GraphsTuple


class RStateFn(nn.Module):
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]
    n_out: int = 1
    rnn_cls: Type[RNN] = None

    @nn.compact
    def __call__(
            self, graph: GraphsTuple, rnn_state: Array, n_agents: int, *args, **kwargs
    ) -> [Array, Array]:
        """
        rnn_state: (n_layers, n_carries, hid_size)
        """
        x = self.gnn_cls()(graph, node_type=0, n_type=n_agents)

        # aggregate information using mean
        x = x.mean(axis=0, keepdims=True)  # (1, msg_dim)

        # pass through head class
        x = self.head_cls()(x)  # (1, msg_dim)

        # pass through RNN
        if self.rnn_cls is not None:
            x, rnn_state = self.rnn_cls()(x, rnn_state)

        # get value
        x = nn.Dense(self.n_out, kernel_init=default_nn_init())(x)
        assert x.shape == (1, self.n_out)

        return x, rnn_state


class DecRStateFn(nn.Module):
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]
    n_out: int = 1
    rnn_cls: Type[RNN] = None
    use_global_info: bool = False

    @nn.compact
    def __call__(
            self, graph: GraphsTuple, rnn_state: Array, n_agents: int, *args, **kwargs
    ) -> [Array, Array]:
        """
        rnn_state: (n_layers, n_carries, hid_size)
        """
        x = self.gnn_cls()(graph, node_type=0, n_type=n_agents)  # (n_agent, msg_dim)

        if self.use_global_info:
            x_global = x.mean(axis=0, keepdims=True)  # (1, msg_dim)
            x = jnp.concatenate([x, jnp.tile(x_global, (n_agents, 1))], axis=-1)  # (n_agent, 2 * msg_dim)

        # pass through head class
        x = self.head_cls()(x)  # (n_agent, msg_dim)
        assert x.shape[0] == n_agents

        # pass through RNN
        if self.rnn_cls is not None:
            x, rnn_state = self.rnn_cls()(x, rnn_state)

        # get value
        x = nn.Dense(self.n_out, kernel_init=default_nn_init())(x)  # (n_agent, n_out)
        assert x.shape == (n_agents, self.n_out)

        return x, rnn_state


class ValueNet:

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            n_out: int = 1,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            gnn_layers: int = 1,
            gnn_out_dim: int = 16,
            use_lstm: bool = False,
            decompose: bool = False,
            use_global_info: bool = False,
            n_heads: int = 3
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.n_out = n_out
        self.gnn_out_dim = gnn_out_dim
        self.decompose = decompose
        self.use_global_info = use_global_info

        self.gnn = ft.partial(
            GraphTransformerGNN,
            msg_dim=32,
            out_dim=gnn_out_dim,
            n_heads=n_heads,
            n_layers=gnn_layers
        )
        self.head = ft.partial(
            MLP,
            hid_sizes=(64, 64),
            act=nn.relu,
            act_final=True,
            name='ValueGNNHead'
        )

        self.use_rnn = use_rnn
        if use_rnn:
            self.rnn_base = ft.partial(nn.LSTMCell if use_lstm else nn.GRUCell, features=64)
            self.rnn = ft.partial(
                RNN,
                rnn_cls=self.rnn_base,
                rnn_layers=rnn_layers
            )
        else:
            self.rnn = None

        if decompose:
            self.net = DecRStateFn(
                gnn_cls=self.gnn,
                head_cls=self.head,
                n_out=n_out,
                rnn_cls=self.rnn,
                use_global_info=use_global_info,
            )
        else:
            self.net = RStateFn(
                gnn_cls=self.gnn,
                head_cls=self.head,
                n_out=n_out,
                rnn_cls=self.rnn,
            )

    def initialize_carry(self, key: PRNGKey):
        if self.use_rnn:
            return self.rnn_base().initialize_carry(key, (self.gnn_out_dim,))
        else:
            return jnp.zeros((self.gnn_out_dim,))

    def get_value(self, params: Params, obs: GraphsTuple, rnn_state: Array) -> [Array, Array]:
        values, rnn_state = self.net.apply(params, obs, rnn_state, self.n_agents)
        return values, rnn_state
