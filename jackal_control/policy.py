import os
import numpy as np
import yaml
import jax.random as jr
import jax.numpy as jnp

from typing import Tuple

from dgppo.env import make_env
from dgppo.env.lidar_env.base import LidarEnvState
from dgppo.env.lidar_env.lidar_dubins_target import LidarMoveObsEnvState
from dgppo.algo import make_algo
from dgppo.utils.graph import GraphsTuple
from dgppo.utils.typing import Action


class Policy:

    def __init__(
            self,
            path: str,
            key: jr.PRNGKey
    ):
        # params
        self.acc_scale = 10.
        self.omega_scale = 0.4
        self.dt = 0.02

        self.path = path

        # load config
        with open(os.path.join(path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        # make env
        self.env = make_env(
            env_id=config.env,
            num_agents=config.num_agents,
            num_obs=config.obs,
        )
        self.state_lim = self.env.state_lim

        # create algorithm
        self.algo = make_algo(
            algo=config.algo,
            env=self.env,
            node_dim=self.env.node_dim,
            edge_dim=self.env.edge_dim,
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            n_agents=self.env.num_agents,
            cost_weight=config.cost_weight,
            actor_gnn_layers=config.actor_gnn_layers,
            Vl_gnn_layers=config.Vl_gnn_layers,
            Vh_gnn_layers=config.Vh_gnn_layers if hasattr(config, "Vh_gnn_layers") else 1,
            lr_actor=config.lr_actor,
            lr_Vl=config.lr_Vl,
            max_grad_norm=2.0,
            seed=config.seed,
            use_rnn=config.use_rnn,
            rnn_layers=config.rnn_layers,
            use_lstm=config.use_lstm,
        )

        # load model
        model_path = os.path.join(path, "models")
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
        print("step: ", step)
        self.algo.load(model_path, step)

        # create act function
        self.act_fn = self.algo.act
        self.init_rnn_state = self.algo.init_rnn_state

        # get a nominal graph
        graph_key, key = jr.split(key)
        self.nominal_graph = self.env.reset(graph_key)
        self.key = key

    def create_graph(self, jackal_state, goal_pos, human_pos) -> GraphsTuple:
        # goal = self.nominal_graph.type_states(type_idx=1, n_type=1)
        goal = jnp.zeros_like(jackal_state)
        goal = goal.at[:, :2].set(goal_pos)
        human_pos = jnp.zeros_like(jackal_state)
        human_pos = human_pos.at[:, :2].set(human_pos)
        env_state = LidarMoveObsEnvState(jackal_state, goal, None, human_pos)  # currently no obstacles
        return self.env.get_graph(env_state)

    def get_action(self, graph: GraphsTuple) -> Tuple[float, float]:
        # get NN outputs
        u_nn, _ = self.act_fn(graph, self.init_rnn_state)

        # convert to real control
        agent_states = graph.type_states(type_idx=0, n_type=self.env.num_agents)
        omega = u_nn[0, 0]
        acc = u_nn[0, 1] * self.acc_scale
        v = acc * self.dt + agent_states[0, 4]
        v = np.clip(v, self.state_lim()[0][4], self.state_lim()[1][4])

        return omega, v
