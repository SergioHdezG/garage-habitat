#!/usr/bin/env python3
"""This is an example to train MAML-PPO on Maze environment."""
# pylint: disable=no-value-for-parameter
import click
import torch
from gym_miniworld.envs import MazeS3Fast, MazeS3, MazeS5

import os
from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import RaySampler, VecWorker, LocalSampler, CustomLocalSampler, CustomVecWorker
from garage.torch.algos import MAMLPPO, MAMLTRPO
from garage.torch.policies import CategoricalCNNPolicy, ResNetCNNPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.torch import set_gpu_mode
from environments import habitat_envs
import numpy as np
import gym
import akro
import habitat.utils.gym_definitions

@click.command()
@click.option('--seed', default=27)
@click.option('--epochs', default=20)
@click.option('--episodes_per_task', default=1)
@click.option('--meta_batch_size', default=1)
@click.option('--max_episode_length', default=35)
@click.option('--inner_lr', default=0.01)
@click.option('--outer_lr', default=1e-3)
@wrap_experiment(snapshot_mode='all', log_dir='/home/reyjc/resultados/',
                 prefix='experiments',
                 archive_launch_repo=False)
def maml_ppo_resnet_maze(ctxt, seed, epochs, episodes_per_task,
                         meta_batch_size, max_episode_length, inner_lr,
                         outer_lr):
    """Set up environment and algorithm and run the task.

    Args:
        max_episode_length:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        episodes_per_task (int): Number of episodes per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.

    """
    # gym_env = gym.make("HabitatCloseCab-v0")
    set_seed(seed)
    env = GymEnv(habitat_envs.SimpleRLEnvRGB(config_paths="configs/tasks/pointnav.yaml",
                                          result_path=os.path.join("development", "images")),
                                          is_image=True,
                                          max_episode_length=max_episode_length)
    # env = GymEnv(habitat_envs.SimpleRLEnv(),
    #              is_image=True,
    #              max_episode_length=max_episode_length)

    policy = ResNetCNNPolicy(
        env_spec=env.spec,
        hidden_nonlinearity=torch.relu,
        hidden_sizes=(512, 256)
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None,
                                              is_image=True)

    # task_sampler = SetTaskSampler(
    #     habitat_envs.SimpleRLEnv,
    #     wrapper=lambda env, _: GymEnv(env, is_image=True, max_episode_length=max_episode_length))
    task_sampler = SetTaskSampler(env_constructor=habitat_envs.SimpleRLEnvRGB,
                                  env=env,
                                  wrapper=lambda env, _: GymEnv(env, is_image=True, max_episode_length=max_episode_length))

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   n_test_tasks=meta_batch_size,
                                   n_test_episodes=episodes_per_task,
                                   n_exploration_eps=episodes_per_task,
                                   # worker_class=VecWorker,
                                   # worker_args=dict(n_envs=1),
                                   )

    trainer = Trainer(ctxt)

    # sampler = CustomLocalSampler(agents=policy,
    #                      envs=env,
    #                      worker_class=CustomVecWorker,
    #                      worker_args=dict(n_envs=2, deepcopy=habitat_envs.SimpleRLEnv_deepcopy),
    #                      max_episode_length=env.spec.max_episode_length,
    #                      deepcopy=habitat_envs.SimpleRLEnv_deepcopy)
    sampler = LocalSampler(agents=policy,
                         envs=env,
                         # worker_class=VecWorker,
                         # worker_args=dict(n_envs=1),
                         max_episode_length=env.spec.max_episode_length)

    algo = MAMLPPO(env=env,
                   policy=policy,
                   sampler=sampler,
                   task_sampler=task_sampler,
                   value_function=value_function,
                   meta_batch_size=meta_batch_size,
                   gae_lambda=1.,
                   inner_lr=inner_lr,
                   outer_lr=outer_lr,
                   num_grad_updates=1,
                   meta_evaluator=meta_evaluator,
                   evaluate_every_n_epochs=5)

    # send policy to GPU
    if torch.cuda.is_available():
        device = set_gpu_mode(True)
        policy.to(device=device)

    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=episodes_per_task,
                  store_episodes=True)  # batch size no more than
    # 400 or 500 aprox due to RAM limitations (128GB)


maml_ppo_resnet_maze()
