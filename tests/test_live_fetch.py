"""The optional live-acquisition client -- cache, rate gate, hard-stop.

No network: a fake transport is injected. Confirms the §5.1 behaviours that
matter (cache-and-freeze, backoff on soft failure, hard stop on a block).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from courtgraph.ingest.live_fetch import (  # noqa: E402
    LiveAccessBlocked,
    LiveAccessError,
    LiveCache,
    LiveClient,
    _Clock,
)


class _NoWaitClock(_Clock):
    def wait(self) -> None:  # never actually sleep in tests
        return None


class LiveFetchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = LiveCache(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fetches_once_then_serves_from_cache(self) -> None:
        calls: list[tuple[str, dict[str, str]]] = []

        def transport(endpoint: str, params: dict[str, str]) -> dict[str, object]:
            calls.append((endpoint, dict(params)))
            return {"resultSets": [], "params": params}

        client = LiveClient(self.cache, transport=transport, clock=_NoWaitClock())
        a = client.fetch("boxscoretraditionalv2", {"GameID": "0022000001"})
        b = client.fetch("boxscoretraditionalv2", {"GameID": "0022000001"})
        self.assertEqual(a, b)
        self.assertEqual(len(calls), 1)  # second call served from disk

        # a fresh client re-reads the same on-disk cache
        again = LiveClient(
            LiveCache(Path(self._tmp.name)), transport=transport, clock=_NoWaitClock()
        )
        again.fetch("boxscoretraditionalv2", {"GameID": "0022000001"})
        self.assertEqual(len(calls), 1)

    def test_hard_stop_on_block_is_not_retried(self) -> None:
        attempts = 0

        def transport(endpoint: str, params: dict[str, str]) -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            raise LiveAccessBlocked("HTTP 403")

        client = LiveClient(self.cache, transport=transport, clock=_NoWaitClock())
        with self.assertRaises(LiveAccessBlocked):
            client.fetch("commonteamroster", {"TeamID": "1610612744"})
        self.assertEqual(attempts, 1)  # no retry on a hard block

    def test_soft_error_is_retried_then_gives_up(self) -> None:
        attempts = 0

        def transport(endpoint: str, params: dict[str, str]) -> dict[str, object]:
            nonlocal attempts
            attempts += 1
            raise LiveAccessError("HTTP 503")

        client = LiveClient(self.cache, transport=transport, clock=_NoWaitClock())
        with (
            mock.patch("courtgraph.ingest.live_fetch.time.sleep"),
            self.assertRaises(LiveAccessError),
        ):
            client.fetch("commonteamroster", {"TeamID": "1610612744"})
        self.assertGreater(attempts, 1)

    def test_cache_survives_transport_going_away(self) -> None:
        def transport(endpoint: str, params: dict[str, str]) -> dict[str, object]:
            return {"ok": True}

        LiveClient(self.cache, transport=transport, clock=_NoWaitClock()).fetch(
            "x", {"a": "1"}
        )

        def dead(endpoint: str, params: dict[str, str]) -> dict[str, object]:
            raise AssertionError("should not be called")

        got = LiveClient(
            LiveCache(Path(self._tmp.name)), transport=dead, clock=_NoWaitClock()
        ).fetch("x", {"a": "1"})
        self.assertEqual(got, {"ok": True})


if __name__ == "__main__":
    unittest.main()
