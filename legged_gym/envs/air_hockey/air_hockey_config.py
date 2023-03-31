from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AirHockeyCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class asset_table(LeggedRobotCfg.asset):
        file = ''
        name = 'table'

    class asset_robot(LeggedRobotCfg.asset):
        file = ''
        name = 'robot'


class AirHockeyCfgPPO(LeggedRobotCfgPPO):
    pass
