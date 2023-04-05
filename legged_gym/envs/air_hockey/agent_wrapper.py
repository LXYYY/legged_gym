import os
import numpy as np


def get_parent_directory(module_name):
    import importlib
    # Load the module
    module = importlib.import_module(module_name)

    # Get the module's file path
    module_path = os.path.abspath(module.__file__)

    # Find the directory containing the module
    module_dir = os.path.dirname(module_path)

    # Find the parent directory
    parent_dir = os.path.abspath(os.path.join(module_dir, '..'))

    return parent_dir


pdir = get_parent_directory("air_hockey_challenge")
import sys

sys.path.append(pdir)
from examples.control.hitting_agent import HittingAgent


class AgentWrapper:
    def __init__(self, env_info, num_envs, agent_class=HittingAgent):
        self.agents = [agent_class(env_info) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.reset()

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, obs):
        obs = obs.view(self.num_envs, -1).detach().cpu().numpy()
        actions = [agent.draw_action(obs[i]) for i, agent in enumerate(self.agents)]
        actions = np.array(actions)
        return actions
