from .air_hockey_config import AirHockeyCfg, AirHockeyCfgPPO

import pkg_resources


class AirHockeyPlanarCfg(AirHockeyCfg):
    class asset(AirHockeyCfg.asset):
        file = pkg_resources.resource_filename('air_hockey_challenge', 'environments/data/planar/single_isaac.xml')

    class rewards(AirHockeyCfg.rewards):
        class scales:
            pass

    class control(AirHockeyCfg.control):
        control_type = 'P'
        stiffness = {'planar_robot_1/joint_1': 960, 'planar_robot_1/joint_2': 480, 'planar_robot_1/joint_3': 240}
        damping = {'planar_robot_1/joint_1': 60, 'planar_robot_1/joint_2': 20, 'planar_robot_1/joint_3': 4}


class AirHockeyPlanarCfgPPO(AirHockeyCfgPPO):
    pass
