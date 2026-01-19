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
    """Logging dictionary returned by loss function."""
    loss_base: float
    loss_dice: float
    loss_bce: float
    loss_tear: float
    loss_total: float
    loss_tear_enabled: int