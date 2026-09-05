"""S1a — conformal acceptance gate for loop-closure verification
(split-conformal false-accept control).

A ConformalGate scores a verified candidate by a danger score s(x) = P(false
| features) (L2-penalized logistic on standardized features) and accepts iff
s(x) < Q, where Q is the j-th smallest danger score over n_cal DANGEROUS
calibration candidates with j = floor((n_cal + 1) * alpha). By the
split-conformal quantile lemma this gives the marginal finite-sample
guarantee

    P(accept | candidate is false) <= alpha

under exchangeability of the false-candidate pool — no distributional
assumption on the score. Fit/evaluation protocol and gate verdicts:
scripts/eval_conformal_s1.py (output_s1/s1_report.md). The heuristic
two-metric gate remains the pipeline default; this module is the documented
S1a arm (loop_closure.verify_candidates gate_model=...).

Two serializations ship in data/conformal_gate/:
  A1: features [overlap, mean_dist] — self-contained (needs only RAP
      diagnostics), TEST recall 1.000 / FPR 0.053 (bound 0.05 + 1/(n+1)).
  A2: + [sc_d1, sc_margin, deg_loc_t, deg_loc_r, deg_scat] — the predeclared
      primary arm, TEST recall 0.975 / FPR 0.035; needs the retrieval margin
      and scan-degeneracy features (available in the live node; offline
      verify_candidates has no SC DB, so A2 is live-side only).
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConformalGate:
    def __init__(self, features: List[str], coef: List[float],
                 intercept: float, mu: List[float], sd: List[float],
                 Q: float, alpha: float, n_calib_d: int):
        self.features = list(features)
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.mu = np.asarray(mu, dtype=float)
        self.sd = np.asarray(sd, dtype=float)
        self.Q = float(Q)
        self.alpha = float(alpha)
        self.n_calib_d = int(n_calib_d)

    # ------------------------------------------------------------------
    def score(self, feats: Dict[str, float]) -> Optional[float]:
        """Danger score P(false | features); None when a feature is missing.
        """
        v = []
        for f in self.features:
            x = feats.get(f)
            if x is None or not np.isfinite(x):
                return None
            v.append(float(x))
        z = (np.asarray(v) - self.mu) / self.sd
        logit = float(z @ self.coef + self.intercept)
        return 1.0 / (1.0 + np.exp(-logit))

    def accept(self, feats: Dict[str, float]) -> Optional[bool]:
        s = self.score(feats)
        return None if s is None else bool(s < self.Q)

    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, records: List[dict], features: List[str],
            alpha: float = 0.05):
        """Fit on ALL labeled records (deployment model; the evaluation used
        split-conformal — for deployment every dangerous sample calibrates).
        records: dicts with features + "label" in {"safe", "dangerous"}.
        """
        from sklearn.linear_model import LogisticRegression
        recs = [r for r in records
                if r.get("label") in ("safe", "dangerous")
                and all(r.get(f) is not None and np.isfinite(r.get(f))
                        for f in features)]
        X = np.stack([[float(r[f]) for f in features] for r in recs])
        y = np.array([1 if r["label"] == "dangerous" else 0 for r in recs])
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        clf = LogisticRegression(C=1.0, max_iter=5000)
        clf.fit((X - mu) / sd, y)
        gate = cls(features, clf.coef_[0].tolist(),
                   float(clf.intercept_[0]), mu.tolist(), sd.tolist(),
                   Q=0.0, alpha=alpha, n_calib_d=0)
        d_scores = sorted(gate.score({f: float(r[f])
                                      for f in features})
                          for r, yy in zip(recs, y) if yy == 1)
        n = len(d_scores)
        j = int(np.floor((n + 1) * alpha))
        gate.Q = d_scores[j - 1] if j > 0 else -np.inf
        gate.n_calib_d = n
        logger.info(f"ConformalGate.fit: {len(recs)} records "
                    f"({n} dangerous), features={features}, Q={gate.Q:.4f}")
        return gate

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"features": self.features, "coef": self.coef.tolist(),
                       "intercept": self.intercept, "mu": self.mu.tolist(),
                       "sd": self.sd.tolist(), "Q": self.Q,
                       "alpha": self.alpha, "n_calib_d": self.n_calib_d},
                      f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConformalGate":
        return cls(**json.load(open(path)))
