"""
skills/__init__.py
Coding-agent skill registry.
"""

from .project_scanner    import SKILL_DEF as SCANNER_DEF
from .code_writer        import SKILL_DEF as WRITER_DEF
from .code_reviewer      import SKILL_DEF as REVIEWER_DEF
from .code_surgeon       import SKILL_DEF as SURGEON_DEF
from .test_runner        import SKILL_DEF as TEST_RUNNER_DEF
from .dependency_resolver import SKILL_DEF as DEP_DEF

REGISTRY: dict[str, dict] = {}


def _register(skill_def: dict):
    REGISTRY[skill_def["name"]] = skill_def


_register(SCANNER_DEF)
_register(WRITER_DEF)
_register(SURGEON_DEF)
_register(REVIEWER_DEF)
_register(TEST_RUNNER_DEF)
_register(DEP_DEF)

# ── Auto-load learned skills ───────────────────────────────────────────────────
import importlib.util
from pathlib import Path

_learned_dir = Path(__file__).parent / "learned"
_learned_dir.mkdir(exist_ok=True)

for _skill_file in sorted(_learned_dir.glob("*.py")):
    if _skill_file.name.startswith("_"):
        continue
    try:
        _spec   = importlib.util.spec_from_file_location(
            f"skills.learned.{_skill_file.stem}", _skill_file
        )
        _module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_module)
        if hasattr(_module, "SKILL_DEF"):
            _register(_module.SKILL_DEF)
    except Exception as _e:
        import sys
        print(f"[skills] Warning: could not load {_skill_file.name}: {_e}", file=sys.stderr)


def get(name: str) -> dict | None:
    return REGISTRY.get(name)


def all_skills() -> list[dict]:
    return list(REGISTRY.values())


def describe_all() -> str:
    lines = []
    for s in REGISTRY.values():
        tag = " [learned]" if s.get("learned") else ""
        lines.append(f"  - skill:{s['name']}: {s['description']}{tag}")
    return "\n".join(lines)
