from .air_hockey_config import AirHockeyCfg, AirHockeyCfgPPO

import pkg_resources


class AirHockeyPlanarCfg(AirHockeyCfg):
    class asset(AirHockeyCfg.asset_robot):
        file = pkg_resources.resource_filename('air_hockey_challenge', 'environments/data/planar/single_isaac.xml')

    class asset_table(AirHockeyCfg.asset_table):
        file = pkg_resources.resource_filename('air_hockey_challenge', 'environments/data/planar/table_isaac.xml')

    class init_state(AirHockeyCfg.init_state):
        default_joint_angles = {
            'planar_robot_1/joint_1': -1.15570723,
            'planar_robot_1/joint_2': 1.30024401,
            'planar_robot_1/joint_3': 1.44280414
        }
        random_joints = {
            'pack_x',
            'pack_y',
            'pack_yaw'
        },
        control_joint_idx={
            'planar_robot_1/joint_1': 0,
            'planar_robot_1/joint_2': 1,
            'planar_robot_1/joint_3': 2
        }
        pos = [0, 0, 0]

    class rewards(AirHockeyCfg.rewards):
        class scales:
            pass

    class control(AirHockeyCfg.control):
        control_type = 'P'
        stiffness = {'planar_robot_1/joint_1': 960, 'planar_robot_1/joint_2': 480, 'planar_robot_1/joint_3': 240}
        damping = {'planar_robot_1/joint_1': 60, 'planar_robot_1/joint_2': 20, 'planar_robot_1/joint_3': 4}


class AirHockeyPlanarCfgPPO(AirHockeyCfgPPO):
    pass
