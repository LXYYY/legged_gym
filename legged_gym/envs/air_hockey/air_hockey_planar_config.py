from .air_hockey_config import AirHockeyCfg, AirHockeyCfgPPO

import pkg_resources


class AirHockeyPlanarCfg(AirHockeyCfg):
    class asset(AirHockeyCfg.asset):
        file = pkg_resources.resource_filename('air_hockey_challenge', 'environments/data/planar/single_isaac.xml')

    class rewards(AirHockeyCfg.rewards):
        class scales:
            pass


class AirHockeyPlanarCfgPPO(AirHockeyCfgPPO):
    pass
