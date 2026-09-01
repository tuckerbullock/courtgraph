"""Rung 4 -- explicit teammate-pair interaction RAPM."""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _chemistry_support import HAS_NUMPY  # noqa: E402

if TYPE_CHECKING:
    from courtgraph.chemistry.features import DesignMatrices, FeatureSpace
    from courtgraph.chemistry.pair_interaction import (
        PairHierarchicalRidge,
        PairVocabulary,
    )
    from courtgraph.chemistry.stints import StintTable


def _pair_effect_dataset(
    *,
    n_players: int = 60,
    n_stints: int = 6000,
    tau_off: float = 3.0,
    tau_def: float = 2.5,
    tau_pair: float = 2.0,
    sigma: float = 9.0,
    seed: int = 7,
) -> tuple[StintTable, dict[str, float], set[str]]:
    """A well-specified additive + free-per-pair world: no low-rank structure.

    ``y_s = 110 + Σα_i − Σβ_j + Σγ_ij + N(0, σ²/w)`` with a planted
    ``γ_ij ~ N(0, tau_pair²)`` on **every** offensive pair that ever co-occurs.
    Lineups come from 4 rotation groups plus a 12% cross-group draw, so pair
    exposure spans from a handful of stints to hundreds -- some pairs will fall
    below any sensible vocabulary threshold.
    Returns (table, {pair_key: gamma}, all_co_occurring_pair_keys).
    """

    import numpy as np

    from courtgraph.chemistry.stints import Stint, StintTable, pair_id

    rng = np.random.default_rng(seed)
    players = list(range(1000, 1000 + n_players))
    off_talent = {
        p: float(v)
        for p, v in zip(players, rng.normal(0, tau_off, n_players), strict=True)
    }
    def_talent = {
        p: float(v)
        for p, v in zip(players, rng.normal(0, tau_def, n_players), strict=True)
    }

    groups = [players[i * 15 : (i + 1) * 15] for i in range(4)]
    stints: list[Any] = []
    lineups: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for _ in range(n_stints):
        gi, gj, gk = (int(x) for x in rng.choice(4, 3, replace=False))
        off_pool = groups[gi] + groups[gk] if rng.random() < 0.12 else groups[gi]
        off = tuple(sorted(int(x) for x in rng.choice(off_pool, 5, replace=False)))
        deff = tuple(sorted(int(x) for x in rng.choice(groups[gj], 5, replace=False)))
        lineups.append((off, deff))

    co: dict[str, int] = {}
    for off, _deff in lineups:
        for a in range(5):
            for b in range(a + 1, 5):
                key = pair_id(off[a], off[b])
                co[key] = co.get(key, 0) + 1
    admitted = set(co)
    gamma = {k: float(rng.normal(0, tau_pair)) for k in admitted}

    for i, (off, deff) in enumerate(lineups):
        w = int(rng.integers(4, 16))
        value = (
            110.0
            + sum(off_talent[p] for p in off)
            - sum(def_talent[p] for p in deff)
            + sum(
                gamma.get(pair_id(off[a], off[b]), 0.0)
                for a in range(5)
                for b in range(a + 1, 5)
            )
        )
        y = value + float(rng.normal(0, sigma / np.sqrt(w)))
        stints.append(
            Stint(
                stint_id=f"s{i}",
                game_id=f"g{i // 10}",
                game_date=f"2021-{1 + (i // 400) % 12:02d}-{1 + i % 27:02d}",
                season="2020-21",
                season_index=0,
                period=1 + i % 4,
                start_time_seconds=float(i % 600),
                offense_team_id=1 + (min(off) - 1000) // 15,
                defense_team_id=1 + (min(deff) - 1000) // 15,
                offense_player_ids=off,  # type: ignore[arg-type]
                defense_player_ids=deff,  # type: ignore[arg-type]
                offensive_possessions=w,
                points_scored=int(round(y * w / 100.0)),
                home_offense=bool(i % 2),
                score_margin_offense=0,
                playoff=False,
                days_rest_offense=1 + i % 3,
                garbage_time_weight=1.0,
            )
        )
    return StintTable.from_stints(stints), gamma, admitted


@unittest.skipUnless(HAS_NUMPY, "rung 4 requires numpy")
class PairGramEquivalenceTests(unittest.TestCase):
    def test_new_blocks_match_a_dense_one_hot_reference(self) -> None:
        import numpy as np

        from courtgraph.chemistry.baseline import _cross_ctx, _cross_gram, _cross_rhs

        rng = np.random.default_rng(3)
        n, p, q = 500, 12, 40
        offense_index = np.stack([rng.permutation(p)[:5] for _ in range(n)]).astype(
            np.int64
        )
        # 10 distinct admitted-pair rows per stint (mirrors real data: the 10
        # slot pairs of 5 distinct players are 10 distinct pairs), ~30% -> -1
        opair = np.stack([rng.permutation(q)[:10] for _ in range(n)]).astype(np.int64)
        opair[rng.random((n, 10)) < 0.3] = -1
        w = rng.uniform(1.0, 9.0, size=n)
        ctx = rng.normal(size=(n, 4))
        cw = ctx * w[:, None]

        def onehot(index: np.ndarray, dim: int) -> np.ndarray:
            m = np.zeros((n, dim))
            for r in range(n):
                for v in index[r]:
                    if v >= 0:
                        m[r, v] = 1.0
            return m

        pair_oh, off_oh = onehot(opair, q), onehot(offense_index, p)
        self.assertTrue(
            np.allclose(
                _cross_gram(opair, opair, w, q, q),
                pair_oh.T @ (w[:, None] * pair_oh),
                atol=1e-8,
            )
        )
        self.assertTrue(
            np.allclose(
                _cross_gram(offense_index, opair, w, p, q),
                off_oh.T @ (w[:, None] * pair_oh),
                atol=1e-8,
            )
        )
        self.assertTrue(
            np.allclose(_cross_ctx(opair, cw, q), pair_oh.T @ cw, atol=1e-8)
        )
        y = rng.normal(size=n)
        self.assertTrue(
            np.allclose(_cross_rhs(opair, w * y, q), pair_oh.T @ (w * y), atol=1e-8)
        )


@unittest.skipUnless(HAS_NUMPY, "rung 4 requires numpy")
class PairHierarchicalRidgeTests(unittest.TestCase):
    table: StintTable
    gamma: dict[str, float]
    admitted: set[str]
    space: FeatureSpace
    design: DesignMatrices
    vocab: PairVocabulary
    model: PairHierarchicalRidge

    @classmethod
    def setUpClass(cls) -> None:
        from courtgraph.chemistry.features import FeatureSpace
        from courtgraph.chemistry.pair_interaction import (
            PairHierarchicalConfig,
            PairHierarchicalRidge,
            PairVocabulary,
        )

        cls.table, cls.gamma, cls.admitted = _pair_effect_dataset()
        cls.space = FeatureSpace.from_training(cls.table)
        cls.design = cls.space.build(cls.table)
        cls.vocab = PairVocabulary.from_training(cls.table, min_co_stints=15)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cls.model = PairHierarchicalRidge.fit(
                cls.design,
                cls.space,
                cls.vocab,
                config=PairHierarchicalConfig(tol=1e-6, max_iters=100),
            )

    def test_recovers_tau_pair(self) -> None:
        import numpy as np

        realised = float(
            np.std([self.gamma[k] for k in self.vocab.pair_ids if k in self.gamma])
        )
        vc = self.model.variance_components()
        self.assertLess(abs(vc["tau_pair"] / realised - 1.0), 0.20)
        self.assertTrue(self.model.converged)

    def test_pair_coefs_correlate_with_the_planted_gamma(self) -> None:
        import numpy as np

        est = np.array([self.model.pair_coef[i] for i in range(self.vocab.n_pairs)])
        tru = np.array([self.gamma.get(k, 0.0) for k in self.vocab.pair_ids])
        self.assertGreater(float(np.corrcoef(est, tru)[0, 1]), 0.5)

    def test_inadmissible_pair_has_zero_surplus_and_additive_fallback(self) -> None:
        import numpy as np

        from courtgraph.chemistry.pair_interaction import build_offense_pair_index

        # a stint whose offense contains at least one non-admitted pair
        opair = build_offense_pair_index(
            self.design.offense_index, self.space, self.vocab
        )
        degraded_rows = np.flatnonzero((opair < 0).any(axis=1))
        self.assertGreater(len(degraded_rows), 0)
        r = int(degraded_rows[0])
        # the pair-term contribution for that row is the sum over ONLY admitted pairs
        contrib = float(self.model.pair_coef[opair[r][opair[r] >= 0]].sum())
        # decompose_row's talent minus the additive-only talent == that contribution
        d = self.model.decompose_row(self.design, r)
        add_only = (
            self.model.alpha
            + float(
                np.take(self.model.offense_coef, self.design.offense_index[r])[
                    self.design.offense_index[r] >= 0
                ].sum()
            )
            - float(
                np.take(self.model.defense_coef, self.design.defense_index[r])[
                    self.design.defense_index[r] >= 0
                ].sum()
            )
        )
        self.assertAlmostEqual(d.talent - add_only, contrib, places=6)

    def test_placebo_vocab_is_a_real_non_injective_control(self) -> None:
        import numpy as np

        from courtgraph.chemistry.baseline_ladder import _placebo_vocab
        from courtgraph.chemistry.pair_interaction import (
            PairHierarchicalConfig,
            PairHierarchicalRidge,
        )

        placebo = _placebo_vocab(self.vocab, seed=0)
        self.assertEqual(placebo.pair_ids, self.vocab.pair_ids)
        assert placebo.row_override is not None
        # drawn with replacement -> distinct pairs collide onto shared rows
        self.assertLess(len(set(placebo.row_override)), self.vocab.n_pairs)

        fit = PairHierarchicalRidge.fit(
            self.design,
            self.space,
            placebo,
            config=PairHierarchicalConfig(tol=1e-6, max_iters=100),
        )
        # the placebo cannot reproduce the real per-pair coefficients: its
        # in-sample fit to the planted signal is materially worse
        real_rss = float(np.sum(self.model.residuals(self.design) ** 2))
        placebo_rss = float(np.sum(fit.residuals(self.design) ** 2))
        self.assertGreater(placebo_rss, real_rss * 1.05)

    def test_is_deterministic_and_round_trips(self) -> None:
        import numpy as np

        from courtgraph.chemistry.pair_interaction import (
            PairHierarchicalConfig,
            PairHierarchicalRidge,
        )

        again = PairHierarchicalRidge.fit(
            self.design,
            self.space,
            self.vocab,
            config=PairHierarchicalConfig(tol=1e-6, max_iters=100),
        )
        self.assertTrue(np.array_equal(again.pair_coef, self.model.pair_coef))
        restored = PairHierarchicalRidge.from_dict(self.model.to_dict())
        self.assertTrue(
            np.allclose(restored.predict(self.design), self.model.predict(self.design))
        )
        self.assertEqual(restored.to_dict(), self.model.to_dict())

    def test_decomposition_identity(self) -> None:
        d = self.model.decompose_row(self.design, 0)
        self.assertAlmostEqual(d.talent + d.context, d.total, places=9)


if __name__ == "__main__":
    unittest.main()
