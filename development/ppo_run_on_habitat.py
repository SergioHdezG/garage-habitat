#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import click
import torch
import random
import copy
import os

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler, LocalSampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from environments import habitat_envs
from datetime import datetime
from garage.torch.optimizers import OptimizerWrapper

@click.command()
@click.option('--seed', default=datetime.now())
@click.option('--epochs', default=20)
@click.option('--num_train_per_epoch', default=20)
@click.option('--batch_size', default=2)
@click.option('--max_episode_length', default=10)
@click.option('--n_workers', default=1)
@click.option('--lr', default=1e-10)
@wrap_experiment(log_dir='/home/reyjc/resultados/',
                 prefix='experiments',
                 archive_launch_repo=False)
def ppo_on_habitat(ctxt, seed, epochs, num_train_per_epoch, batch_size, max_episode_length, n_workers, lr):
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
        batch_size (int): Number of tasks sampled per batch.

    """
    random.seed(seed)
    seed = random.randint(0, 2**64)
    set_seed(seed)
    env = GymEnv(habitat_envs.HM3DRLEnv(),
                 is_image=True,
                 max_episode_length=max_episode_length)

    policy = GaussianMLPPolicy(copy.deepcopy(env.spec),
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    trainer = Trainer(ctxt)

    # sampler = RaySampler(agents=policy,
    #                      envs=env,
    #                      max_episode_length=env.spec.max_episode_length)

    sampler = LocalSampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length,
                         n_workers=1)

    algo = PPO(env_spec=copy.deepcopy(env.spec),
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               # policy_optimizer=OptimizerWrapper(lambda lr: torch.optim.Adam(lr), policy),
               # vf_optimizer=OptimizerWrapper(torch.optim.Adam, value_function),
               lr_clip_range=0.2,
               num_train_per_epoch=num_train_per_epoch,
               discount=0.99,
               gae_lambda=0.97,
               center_adv=True,
               positive_adv=False,
               policy_ent_coeff=0.0,
               use_softplus_entropy=False,
               stop_entropy_gradient=False,
               entropy_method='no_entropy')

    # del env  # elimino el entornoj de memoria porque ya no hace falta # hace falta que se mantenga las variables de contxto de openGL, no se puede borrar sin tener ciertas cosas en cuenta.

    # trainer.setup(algo, env)
    trainer.setup(algo, None)  # quizás así no funcione al intentar renderizar el entorno para ver las ejecuciones ya
    # que env se usa en get_env_copy() que se llama desde _start_worker en la intrucción de plotting

    trainer.train(n_epochs=epochs, batch_size=batch_size)

info = {
    'log_dir': '/home/reyjc/resultados/',
    'use_existing_dir': True}

ppo_on_habitat()
