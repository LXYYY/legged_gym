from .air_hockey_config import AirHockeyCfg, AirHockeyCfgPPO

import pkg_resources


class AirHockeyPlanarCfg(AirHockeyCfg):
    class asset(AirHockeyCfg.asset):
        file = pkg_resources.resource_filename('air_hockey_challenge', '/environments/data/planar/single.xml')
        name = 'single'

    class init_state(AirHockeyCfg.init_state):
        default_joint_angles = {
            'planar_robot_1/joint_1': -1.15570723,
            'planar_robot_1/joint_2': 1.30024401,
            'planar_robot_1/joint_3': 1.44280414
        }

    class rewards(AirHockeyCfg.rewards):
        pass


class AirHockeyPlanarCfgPPO(AirHockeyCfgPPO):
    pass
