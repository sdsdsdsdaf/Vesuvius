from typing import Optional, TypedDict

from topometrics import LeaderboardReport


class MetricsResult(TypedDict):
    dice_fg1: float
    precision_fg1: float
    recall_fg1: float
    f1_fg1: float
    leaderboard_score: Optional[LeaderboardReport]
    split_stats: Optional[dict]
    merge_stats: Optional[dict]
    split_merge_proxy: Optional[float]    

class LossLogOutput(TypedDict):
    """Logging dictionary returned by loss function.
    
    loss_total: float\n
    loss_base: float\n
    loss_dice: float\n
    loss_bce: float\n
    loss_tear: float\n
    loss_hd: float\n
    loss_aux: float\n
    hd_enabled: int\n
    aux_enabled: int\n
    aux_type: str\n
    loss_tear_enabled: int
    
    """
    loss_total: float
    loss_base: float
    loss_dice: float
    loss_bce: float
    loss_tear: float
    loss_hd: float
    loss_aux: float
    hd_enabled: int
    aux_enabled: int
    aux_type: str
    loss_tear_enabled: int