"""Deterministic health checks for a CourtGraph development checkout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

from courtgraph import __version__

MINIMUM_PYTHON = (3, 11)
REQUIRED_PROJECT_PATHS = (
    Path("README.md"),
    Path("pyproject.toml"),
    Path("docs/MASTER_PLAN.md"),
    Path("docs/PROJECT_STATUS.md"),
)


@dataclass(frozen=True, slots=True)
class HealthCheck:
    """One machine- and human-readable project health result."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""

        return asdict(self)


def check_python_version(
    version_info: tuple[int, int, int] | None = None,
) -> HealthCheck:
    """Check that the interpreter meets CourtGraph's supported minimum."""

    current = version_info or (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    passed = current[:2] >= MINIMUM_PYTHON
    required = ".".join(str(part) for part in MINIMUM_PYTHON)
    actual = ".".join(str(part) for part in current)
    return HealthCheck(
        name="python_version",
        passed=passed,
        detail=f"Python {actual}; requires Python {required}+",
    )


def check_project_layout(
    project_root: Path,
    required_paths: Iterable[Path] = REQUIRED_PROJECT_PATHS,
) -> HealthCheck:
    """Check that the checkout contains the project's governing files."""

    root = project_root.expanduser().resolve()
    missing = [str(path) for path in required_paths if not (root / path).is_file()]
    if missing:
        return HealthCheck(
            name="project_layout",
            passed=False,
            detail="Missing required files: " + ", ".join(sorted(missing)),
        )
    return HealthCheck(
        name="project_layout",
        passed=True,
        detail=f"Required project files found under {root}",
    )


def run_health_checks(project_root: Path) -> dict[str, Any]:
    """Run all bootstrap checks and return a versioned result document."""

    checks = (
        check_python_version(),
        check_project_layout(project_root),
    )
    healthy = all(check.passed for check in checks)
    return {
        "schema_version": 1,
        "courtgraph_version": __version__,
        "status": "healthy" if healthy else "unhealthy",
        "python_implementation": platform.python_implementation(),
        "checks": [check.to_dict() for check in checks],
    }
