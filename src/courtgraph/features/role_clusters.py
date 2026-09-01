"""Offensive-role clusters from the per-(player, season) profiles.

The role-conditioned interaction model (candidate idea #1) needs a small,
pooled label per player rather than a free per-identity parameter. This module
turns the continuous profile into a standardized role vector and a hard cluster
assignment by deterministic k-means -- "positions beyond listed position"
(master plan section 21.5): lead ball-handler, wing scorer, movement shooter,
stretch big, rim-running big, and so on.

Offense only, matching the rung-4/5 convention. Players below the profile's
exposure floor (rates are ``None``) are not clustered; a stint containing such
a player simply contributes fewer role pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from courtgraph.features.player_season import PlayerSeasonProfile

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# The offensive-role feature set. Turnover rate is left out (it tracks usage);
# defensive counting stats are weak role signal and excluded for the v1
# offense-only model.
ROLE_FEATURES = (
    "usage",
    "three_rate",
    "rim_rate",
    "assist_per100",
    "ft_rate",
    "oreb_per100",
)
DEFAULT_N_CLUSTERS = 5
ROLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RoleClustering:
    """A fitted role space: which (player, season) cells map to which cluster,
    the standardization, and the cluster centers in standardized space."""

    features: tuple[str, ...]
    n_clusters: int
    mean: FloatArray
    std: FloatArray
    centers: FloatArray  # (n_clusters, n_features), standardized
    assignment: dict[tuple[int, str], int]
    player_cluster: dict[int, int]  # collapsed to one role per player
    player_vector: dict[int, tuple[float, ...]]  # standardized role vector
    seed: int

    def cluster_of(self, player_id: int, season: str) -> int:
        """Cluster index for a (player, season) cell, or ``-1`` when the player
        was below the exposure floor that season (not clustered)."""

        return self.assignment.get((player_id, season), -1)

    def role_of(self, player_id: int) -> int:
        """The player's single collapsed role (their highest-exposure season's
        cluster), or ``-1`` if never clustered."""

        return self.player_cluster.get(player_id, -1)

    def vector_of(self, player_id: int) -> tuple[float, ...] | None:
        """The player's standardized role vector (same order as ``features``),
        or ``None`` if never clustered."""

        return self.player_vector.get(player_id)

    def center_profile(self, cluster: int) -> dict[str, float]:
        """The cluster center back in raw (un-standardized) feature units."""

        raw = self.centers[cluster] * self.std + self.mean
        return dict(zip(self.features, (float(x) for x in raw), strict=True))


def _feature_matrix(
    profiles: list[PlayerSeasonProfile], features: tuple[str, ...]
) -> tuple[list[tuple[int, str]], FloatArray]:
    keys: list[tuple[int, str]] = []
    rows: list[list[float]] = []
    for p in profiles:
        values = [getattr(p, name) for name in features]
        if any(v is None for v in values):
            continue
        keys.append((p.player_id, p.season))
        rows.append([float(v) for v in values])
    return keys, np.asarray(rows, dtype=np.float64).reshape(-1, len(features))


def _kmeans(
    data: FloatArray, k: int, *, seed: int, iters: int = 100
) -> tuple[FloatArray, IntArray]:
    """Deterministic Lloyd's algorithm with k-means++ seeding."""

    rng = np.random.default_rng(seed)
    n = data.shape[0]
    # k-means++ init
    centers = [data[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([np.sum((data - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(data[rng.choice(n, p=probs)])
    cen = np.asarray(centers, dtype=np.float64)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        dist = np.sum((data[:, None, :] - cen[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            members = data[labels == j]
            if len(members):
                cen[j] = members.mean(axis=0)
    # stable cluster ids: order by the first feature (usage) of the center
    order = np.argsort(cen[:, 0])
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    return cen[order], remap[labels]


def fit_role_clusters(
    profiles: list[PlayerSeasonProfile],
    *,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    features: tuple[str, ...] = ROLE_FEATURES,
    seed: int = 0,
) -> RoleClustering:
    keys, raw = _feature_matrix(profiles, features)
    if len(keys) < n_clusters:
        raise ValueError(
            f"only {len(keys)} profiles above the exposure floor; need >= "
            f"{n_clusters} to fit {n_clusters} role clusters"
        )
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std == 0.0] = 1.0
    standardized = (raw - mean) / std
    centers, labels = _kmeans(standardized, n_clusters, seed=seed)
    assignment = {key: int(lbl) for key, lbl in zip(keys, labels, strict=True)}

    # collapse to one role per player: the cluster / vector of their
    # highest-exposure season (off_possessions), so a stint-level lookup needs
    # only player id.
    best_poss: dict[int, int] = {}
    player_cluster: dict[int, int] = {}
    player_vector: dict[int, tuple[float, ...]] = {}
    poss = {(p.player_id, p.season): p.off_possessions for p in profiles}
    for i, ((pid, season), lbl) in enumerate(zip(keys, labels, strict=True)):
        exposure = poss.get((pid, season), 0)
        if exposure >= best_poss.get(pid, -1):
            best_poss[pid] = exposure
            player_cluster[pid] = int(lbl)
            player_vector[pid] = tuple(float(x) for x in standardized[i])

    return RoleClustering(
        features=features,
        n_clusters=n_clusters,
        mean=mean,
        std=std,
        centers=centers,
        assignment=assignment,
        player_cluster=player_cluster,
        player_vector=player_vector,
        seed=seed,
    )


def permuted_clustering(clustering: RoleClustering, seed: int) -> RoleClustering:
    """Placebo: the same cluster sizes, but the (player, season) -> cluster map
    is reshuffled. A role-conditioned interaction fit on this cannot carry any
    real role signal; the real model must beat it."""

    rng = np.random.default_rng(seed)
    players = list(clustering.player_cluster)
    perm = rng.permutation(len(players))
    permuted_cluster = {
        players[i]: clustering.player_cluster[players[j]] for i, j in enumerate(perm)
    }
    permuted_vector = {
        players[i]: clustering.player_vector[players[j]]
        for i, j in enumerate(perm)
        if players[j] in clustering.player_vector
    }
    return RoleClustering(
        features=clustering.features,
        n_clusters=clustering.n_clusters,
        mean=clustering.mean,
        std=clustering.std,
        centers=clustering.centers,
        assignment=dict(clustering.assignment),
        player_cluster=permuted_cluster,
        player_vector=permuted_vector,
        seed=seed,
    )
