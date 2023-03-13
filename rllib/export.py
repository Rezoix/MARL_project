import argparse
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.utils.test_utils import check_learning_achieved

from env import Env

from torch import nn
import torch.onnx
import io
import numpy as np
import onnx


parser = argparse.ArgumentParser()

parser.add_argument(
    "--file-name",
    type=str,
    default="C:\\Users\\Saku\\ml-agents-release_20\\build\\UnityEnvironment.exe",
    help="The Unity3d binary (compiled) game, e.g. "
    "'/home/ubuntu/soccer_strikers_vs_goalie_linux.x86_64'. Use `None` for "
    "a currently running Unity3D editor.",
)
parser.add_argument(
    "--path",
    type=str,
    default="C:\\Users\\Saku\\ray_results\\PPO\\PPO_unity3d_9dc5f_00000_0_2023-03-01_01-14-14\\checkpoint_000545",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=3000,
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.",
)


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
        .training(
            lr=0.001,  # 0.0003,
            lambda_=0.95,
            gamma=0.99,
            sgd_minibatch_size=256,
            train_batch_size=4000,
            num_sgd_iter=20,
            clip_param=0.2,
            model={"fcnet_hiddens": [80, 80]},
        )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    )

    trainer = PPOTrainer(config=config)
    trainer.restore(args.path)
    trainer.get_policy("BluePlayer").export_model("./models", onnx=12)

    torchmodel = onnx.load("./models/model.onnx")  # the rllib output model dir

    onnx.checker.check_model(torchmodel)

    graph = torchmodel.graph
    graph.input.pop()  # remove an unused input
    graph.input[0].name = "obs_0"  # rename input
    graph.node[0].input[0] = "obs_0"

    for node in graph.node:
        if node.name == "Identity_13":
            graph.node.remove(node)

    # slice the first half array as true action
    starts = onnx.helper.make_tensor("starts", onnx.TensorProto.INT64, [1], [0])
    ends = onnx.helper.make_tensor("ends", onnx.TensorProto.INT64, [1], [2])
    axes = onnx.helper.make_tensor(
        "axes", onnx.TensorProto.INT64, [1], [-1]
    )  # the last dimention
    graph.initializer.append(starts)
    graph.initializer.append(ends)
    graph.initializer.append(axes)

    # some useless output in inference
    version_number = onnx.helper.make_tensor(
        "version_number", onnx.TensorProto.INT64, [1], [3]
    )
    memory_size = onnx.helper.make_tensor(
        "memory_size", onnx.TensorProto.INT64, [1], [0]
    )
    continuous_action_output_shape = onnx.helper.make_tensor(
        "continuous_action_output_shape", onnx.TensorProto.INT64, [1], [2]
    )
    graph.initializer.append(version_number)
    graph.initializer.append(memory_size)
    graph.initializer.append(continuous_action_output_shape)

    # add the slice node
    node = onnx.helper.make_node(
        "Slice",
        inputs=["output", "starts", "ends", "axes"],
        outputs=["continuous_actions"],
    )
    graph.node.append(node)  # add node in the last layer

    # clear old output and add new output
    while len(graph.output):
        graph.output.pop()
    actions_info = onnx.helper.make_tensor_value_info(
        "continuous_actions", onnx.TensorProto.FLOAT, shape=[]
    )
    graph.output.append(actions_info)
    version_number_info = onnx.helper.make_tensor_value_info(
        "version_number", onnx.TensorProto.INT64, shape=[]
    )
    graph.output.append(version_number_info)
    memory_size_info = onnx.helper.make_tensor_value_info(
        "memory_size", onnx.TensorProto.INT64, shape=[]
    )
    graph.output.append(memory_size_info)
    continuous_action_output_shape_info = onnx.helper.make_tensor_value_info(
        "continuous_action_output_shape", onnx.TensorProto.INT64, shape=[]
    )
    graph.output.append(continuous_action_output_shape_info)

    onnx.checker.check_model(torchmodel)
    onnx.save(
        torchmodel,
        "C:\\Users\\Saku\\ml-agents-release_20\\rllib\\models\\mlagentmodel.onnx",
    )  # save model dir; you can also check your model output in python with onnxruntime
