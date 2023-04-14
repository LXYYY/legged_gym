# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import torch

from legged_gym.envs.air_hockey.agent_wrapper import AgentWrapper

render_only = True


def test_env(args):
    args.task = "air_hockey_planar"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 3)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
    env_wp = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position_velocity", interpolation_order=3,
                                       debug=False)
    env.clone_mujoco_controller(env_wp.base_env)
    env.clone_env_info(env_wp.base_env)

    agent = AgentWrapper(env_wp.base_env.env_info, env.num_envs)

    obs, _ = env.reset()
    vel = [4,4]

    i = 0
    env_ids = torch.arange(3, device='cpu')
    set_vel = True
    if render_only:
        while True:
            if set_vel:
                env.set_puck_vel(env_ids, vel)
                set_vel = False
            env.simulate_step()
            env.render()
            i += 1
            if i > 1000:
                env.reset_idx(env_ids)
                i = 0
                set_vel = True
    else:
        import time
        elapsed_time = 0
        for i in range(int(10 * env.max_episode_length)):
            start_time = time.time()
            actions = None
            if not torch.isnan(obs).any():
                actions = agent.act(obs)
                actions = torch.from_numpy(actions).float().to(obs.device).view(env.num_envs, -1)
            obs, _, rew, done, info = env.step(actions)
            elapsed_time += time.time() - start_time
            if i % 10 == 0:
                # print("frame rate: ", 1.0 / elapsed_time)
                elapsed_time = 0
    print("Done")


if __name__ == '__main__':
    args = get_args()
    test_env(args)
