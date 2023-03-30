from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AirHockeyCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'


class AirHockeyCfgPPO(LeggedRobotCfgPPO):
    pass
