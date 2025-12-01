OWNER_NONE = 0
OWNER_ME   = 1
OWNER_OPP  = 2

class VoronoiInfo:
    """
    Data container for Voronoi results.
    """
    __slots__ = (
        "my_owned", "opp_owned", "contested",
        "max_contested_dist", "min_contested_dist", "min_egg_dist",
        "my_voronoi", "opp_voronoi", "vor_score",
        "contested_up", "contested_right", "contested_down", "contested_left",
        "frag_score",
    )

    def __init__(
        self,
        my_owned, opp_owned, contested,
        max_contested_dist, min_contested_dist, min_egg_dist,
        my_voronoi, opp_voronoi,
        contested_up, contested_right, contested_down, contested_left,
        frag_score,
    ):
        self.my_owned           = my_owned
        self.opp_owned          = opp_owned
        self.contested          = contested
        self.max_contested_dist = max_contested_dist
        self.min_contested_dist = min_contested_dist
        self.min_egg_dist       = min_egg_dist
        self.my_voronoi         = my_voronoi
        self.opp_voronoi        = opp_voronoi
        self.vor_score          = my_voronoi - opp_voronoi

        self.contested_up       = contested_up
        self.contested_right    = contested_right
        self.contested_down     = contested_down
        self.contested_left     = contested_left

        self.frag_score         = frag_score