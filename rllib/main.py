"""
Example of running an RLlib Trainer against a locally running Unity3D editor
instance (available as Unity3DEnv inside RLlib).
For a distributed cloud setup example with Unity,
see `examples/serving/unity3d_[server|client].py`

To run this script against a local Unity3D engine:
1) Install Unity3D and `pip install mlagents`.

2) Open the Unity3D Editor and load an example scene from the following
   ml-agents pip package location:
   `.../ml-agents/Project/Assets/ML-Agents/Examples/`
   This script supports the `3DBall`, `3DBallHard`, `SoccerStrikersVsGoalie`,
    `Tennis`, and `Walker` examples.
   Specify the game you chose on your command line via e.g. `--env 3DBall`.
   Feel free to add more supported examples here.

3) Then run this script (you will have to press Play in your Unity editor
   at some point to start the game and the learning process):
$ python unity3d_env_local.py --env 3DBall --stop-reward [..]
  [--framework=torch]?
"""

import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import check_learning_achieved

from env import Env
from export import SaveCheckpointCallback

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, default=".\\models\\")
parser.add_argument("--restore", type=str, default=None)
parser.add_argument(
    "--file-name",
    type=str,
    default=None,
    help="The Unity3d binary (compiled) game, e.g. "
    "'/home/ubuntu/soccer_strikers_vs_goalie_linux.x86_64'. Use `None` for "
    "a currently running Unity3D editor.",
)
parser.add_argument(
    "--from-checkpoint",
    type=str,
    default=None,
    help="Full path to a checkpoint file for restoring a previously saved "
    "Trainer state.",
)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=9999, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=10000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=9999.0,
    help="Reward at which we stop training.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=3000,
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.",
)
parser.add_argument("--gpus", type=int, default=1, help="How many GPUs should be used.")

if __name__ == "__main__":
    ray.init()

    args = parser.parse_args()

    tune.register_env(
        "unity3d",
        lambda c: Env(
            file_name=c["file_name"],
            no_graphics=(c["file_name"] is not None),
            episode_horizon=c["episode_horizon"],
        ),
    )

    policies, policy_mapping_fn = Env.get_policy_configs_for_game()

    config = (
        PPOConfig()
        .environment(
            "unity3d",
            env_config={
                "file_name": args.file_name,
                "episode_horizon": args.horizon,
            },
            disable_env_checking=True,
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=args.num_workers if args.file_name else 0,
            rollout_fragment_length=200,
        )
        .training(
            lr=0.0003,
            lambda_=0.95,
            gamma=0.99,
            sgd_minibatch_size=256,
            train_batch_size=(args.horizon + 1) * args.num_workers
            if args.file_name
            else (args.horizon + 1),
            # (args.horizon + 1)* 16* args.num_workers,
            num_sgd_iter=20,
            clip_param=0.2,
            model={"fcnet_hiddens": [80, 80]},
        )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .resources(num_gpus=args.gpus)
    )

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.restore:
        tuner = tune.Tuner.restore(args.restore)
    else:
        tuner = tune.Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=air.RunConfig(
                stop=stop,
                verbose=3,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=5,
                    checkpoint_at_end=True,
                ),
                callbacks=[SaveCheckpointCallback("Player", args.model_path)],
            ),
        )

    # Run the experiment.
    results = tuner.fit()

    # And check the results.
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
