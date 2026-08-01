"""Cluster-persistence gate for streaming loop-candidate retrieval.

Motivation (KITTI 02 live runs, 2026-07-23/31): Scan Context distance
alone does not separate true loops from false hits on repetitive urban
geometry — false SC distances go as low as 0.05, overlapping the true
range (0.03-0.10). What separates them is PERSISTENCE: a true revisit
produces a dense cluster of candidate pairs in (i, j) scan-index space
as the vehicle traverses the revisited stretch (27+ pairs on KITTI 02),
while false hits are mostly isolated (38 of 77 false clusters are
singletons).

The gate accumulates candidates into clusters keyed by Chebyshev
proximity in (i, j) space and decides which pairs enter the
verification queue:

  * proximity-tier candidates queue immediately (odometry-based, cheap,
    needed for the POSS baseline) and count as cluster members;
  * strong SC candidates (score < strong_thresh) queue immediately;
  * immediate queueing is NEVER capped — the max_per_cluster cap and the
    in-cluster spacing NMS apply only to persistence promotions, so a
    false proximity hit can never starve a later true SC stream (this
    exact failure mode was observed in simulation on KITTI 02: two false
    proximity hits filled the cluster's slots and blocked 27 true pairs);
  * other SC candidates are promoted only when their cluster reaches
    min_cluster members — the best unpromoted member is queued, subject
    to the spacing NMS and the max_per_cluster cap.

This keeps single-pass loop closures working (POSS: a proximity hit and
a strong SC hit queue immediately) while repetitive-geometry false hits
never reach the expensive RAP verification unless they persist.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ClusterGate:
    """Decide which streaming loop candidates reach the verification queue.

    Args:
        window: Chebyshev distance in (i, j) scan-index space within
            which a candidate joins an existing cluster.
        min_cluster: cluster member count required before a non-strong
            SC member can be promoted.
        max_per_cluster: maximum PERSISTENCE-promoted pairs per cluster
            (immediate strong/proximity queueing is uncapped).
        spacing: in-cluster NMS for persistence promotions — a promoted
            member must be farther than this (Chebyshev, scan indices)
            from every already-queued member of the same cluster.
        strong_thresh: SC score below which candidates queue immediately.
    """

    def __init__(self, window: int = 50, min_cluster: int = 2,
                 max_per_cluster: int = 2, spacing: int = 25,
                 strong_thresh: float = 0.12):
        self.window = window
        self.min_cluster = min_cluster
        self.max_per_cluster = max_per_cluster
        self.spacing = spacing
        self.strong_thresh = strong_thresh
        # Each cluster: {"members": [(i, j, tier, score)],
        #                "queued": [(i, j)],   # everything sent to the worker
        #                "promoted": int}      # persistence promotions used
        self.clusters: List[Dict] = []

    def _find_cluster(self, i: int, j: int) -> Dict:
        for cl in self.clusters:
            for mi, mj, _, _ in cl["members"][-4:]:
                if abs(i - mi) <= self.window and abs(j - mj) <= self.window:
                    return cl
        cl = {"members": [], "queued": [], "promoted": 0}
        self.clusters.append(cl)
        return cl

    def add(self, i: int, j: int, tier: int, score: float
            ) -> List[Tuple[int, int, int, float]]:
        """Register one candidate pair; returns pairs to enqueue now.

        Args:
            i, j: scan indices with i < j.
            tier: 0 = Scan Context hit, 1 = odometry-proximity hit.
            score: queue priority (lower = earlier); SC distance for
                tier 0, normalized odometry distance for tier 1.

        Returns:
            List of (i, j, tier, score) tuples to enqueue (0-2 entries).
        """
        cl = self._find_cluster(i, j)
        cl["members"].append((i, j, tier, score))
        out: List[Tuple[int, int, int, float]] = []

        def enqueue(m, promoted: bool):
            key = (m[0], m[1])
            if key in cl["queued"]:
                return
            cl["queued"].append(key)
            cl["promoted"] += int(promoted)
            out.append((m[0], m[1], m[2], m[3]))

        me = (i, j, tier, score)
        if tier == 1 or score < self.strong_thresh:
            # Immediate: proximity tier or strong SC hit — never capped.
            enqueue(me, promoted=False)

        # Persistence promotion (independent of how the current member
        # was handled — a proximity arrival can also trigger it): once a
        # cluster is dense enough, queue its best not-yet-queued SC
        # member (spacing NMS).
        if (len(cl["members"]) >= self.min_cluster and
                cl["promoted"] < self.max_per_cluster):
            # Persistence promotion: once a cluster is dense enough,
            # queue its best not-yet-queued SC member (spacing NMS).
            for m in sorted(cl["members"], key=lambda m: m[3]):
                if (m[0], m[1]) in cl["queued"] or m[2] == 1:
                    continue
                if any(abs(m[0] - qi) <= self.spacing and
                       abs(m[1] - qj) <= self.spacing
                       for qi, qj in cl["queued"]):
                    continue
                enqueue(m, promoted=True)
                break
        for i_, j_, tier_, score_ in out:
            logger.debug(
                f"cluster gate queued ({i_}, {j_}) tier={tier_} "
                f"score={score_:.3f} (cluster size {len(cl['members'])}, "
                f"promotions {cl['promoted']}/{self.max_per_cluster})")
        return out
