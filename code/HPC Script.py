# coding: utf-8

# In[ ]:


from typing import List

import ray.rllib as rllib
from ray import tune
import ray

from ray.tune.registry import register_env

import gym

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from ray.rllib.utils.exploration.epsilon_greedy import EpsilonGreedy
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer

from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn import DQN 
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel    
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

# from ray.rllib.env.wrappers.atari_wrappers import 

tf1, tf, tfv = try_import_tf()



# https://github.com/ray-project/ray/blob/master/rllib/examples/custom_keras_model.py
# Eigendefiniertes Q_Model
# class AtariModel(DistributionalQTFModel):
class DualAtariModel(DistributionalQTFModel):
    def __init__(self,
                obs_space,
                action_space,
                num_outputs,
                model_config,
                name,
                **kw):
        
        super(DualAtariModel, self).__init__(
            obs_space, 
            action_space, 
            num_outputs, 
            model_config, 
            name, 
            **kw
        )
#         print(obs_space)
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape,
                                           name="input_layer")
        layer1 = tf.keras.layers.Conv2D(32, 8, strides=4,
                                       name="layer1",
                                       activation=tf.nn.relu,
                                       kernel_initializer=normc_initializer(1.0)
                                       )(self.inputs)
        layer2 = tf.keras.layers.Conv2D(64, 4, strides=2,
                                       name="layer2",
                                       activation=tf.nn.relu,
                                       kernel_initializer=normc_initializer(1.0)
                                       )(layer1)
        layer3 = tf.keras.layers.Conv2D(64, 3, strides=1,
                                       name="layer3",
                                       activation=tf.nn.relu,
                                       kernel_initializer=normc_initializer(1.0)
                                       )(layer2)
        flatten = tf.keras.layers.Flatten()(layer3)
        
        dense = tf.keras.layers.Dense(num_outputs,
                                        activation=tf.nn.relu,
                                        name="dense",
                                        kernel_initializer=normc_initializer(1.0)
                                        )(flatten)
        
        #doesnt output the actual q-values. this is handled in the q-head
        self.base_model = tf.keras.Model(self.inputs, dense)

        
    def forward(self, input_dict, state, seq_lens):
#         print(input_dict["obs"])
        model_out = self.base_model(input_dict["obs"])
        return model_out, state

ModelCatalog.register_custom_model("DualAtariModel", DualAtariModel)


#Generelle Einstellungen
config = DQNConfig().to_dict()

DQNConfigTrainer = {
    #DQN
    "dueling": True,
    "double_q": True,
    "target_network_update_freq": 8000,
    "num_gpus": 0.2,
#     "num_workers": 2,
#     "num_envs_per_worker": 2,
    "rollout_fragment_length": 4, #4 = num_envs_per_worker * rollout_fragment_length
    #Einstellungen für den Algorithmus
    "env": tune.grid_search(["BreakoutNoFrameskip-v4","SpaceInvadersNoFrameskip-v4", "PongNoFrameskip-v4", "StarGunnerNoFrameskip-v4"]),
#     "env": "SpaceInvadersNoFrameskip-v4",
#     "env_config" : {
#         "name": "BreakoutNoFrameskip-v4"
#     },
    "noisy": False,
    "num_atoms": 1,
    "gamma": 0.99,
    "lr": 0.0000625, #Lernrate des TD-Errors
    "adam_epsilon": 0.00015,
    "train_batch_size": 32,
    "hiddens": [512],
    "model": {
        "custom_model": "DualAtariModel",
        #architecture of advantage, value streams
#         "custom_model_config": {"hiddens": [512]}
    },
#     "clip_rewards": True, #Clippen zu -1, 0 und 1
#     "preprocessor_pref": "deepmind",
    #Exploration
    "explore": True,
    "exploration_config": {
        "type": "EpsilonGreedy",
        "epsilon_timesteps": 200000,
        "final_epsilon": 0.01
    },
    "evaluation_interval": int(5e4),
    "evaluation_num_workers": 1,
    "evaluation_config": {
        
    },
    #Kompression
    "compress_observations": True,
    #Replay Buffer
    "replay_buffer_config": {
            "type": "MultiAgentPrioritizedReplayBuffer",
            # Specify prioritized replay by supplying a buffer type that supports
            # prioritization, for example: MultiAgentPrioritizedReplayBuffer.
#             "prioritized_replay": DEPRECATED_VALUE,
            # Size of the replay buffer. Note that if async_updates is set,
            # then each worker will have a replay buffer of this size.
            "capacity": int(1e6),
            "prioritized_replay_alpha": 0.7,
            # Beta parameter for sampling from prioritized replay buffer.
            "prioritized_replay_beta": 0.5,
            # Epsilon to add to the TD errors when updating priorities.
            "prioritized_replay_eps": 1e-6,
            # The number of continuous environment steps to replay at once. This may
            # be set to greater than 1 to support recurrent models.
            "replay_sequence_length": 1,
            # Whether to compute priorities on workers.
            "worker_side_prioritization": False,
        },
}

config.update(DQNConfigTrainer)

#TEST
# del config["model"]
# config["hiddens"] = [512]

analysis = tune.run("DQN",
                    name="Test",
                    stop={"agent_timesteps_total": int(2e7)},
                    config=config,
                    keep_checkpoints_num=60,
                    checkpoint_freq=int(1e6),
                    checkpoint_at_end=True,
                    num_samples=1,
                    local_dir="/users/stud/c/chau0000/Scripts/results"
                   )

