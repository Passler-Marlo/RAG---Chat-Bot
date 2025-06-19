"""Minimal wrapper around Meta's Large Concept Model (LCM).

This script acts as a tiny discourse planner.  Given a query it returns a small
list of concept sentences produced by LCM.  The resulting plan can then be fed
into a surface realiser.
"""

from __future__ import annotations

from typing import List


def _load_models():
    """Load LCM and SONAR models or raise a helpful error message."""
    try:
        from lcm import LCM
        from sonar import SonarModel
    except ImportError as exc:  # pragma: no cover - optional deps
        raise SystemExit(
            "LCM and SonarModel libraries are required: {}".format(exc)
        ) from exc

    lcm = LCM.from_pretrained("base_lcm")
    sonar = SonarModel.from_pretrained("sonar_all")
    return lcm, sonar


def generate_concept_plan(
    query: str,
    *,
    max_concepts: int = 4,
    temperature: float = 0.4,
) -> List[str]:
    """Generate a list of concept sentences for ``query``."""

    lcm, sonar = _load_models()

    concept_vec = sonar.encode([query])
    concept_plan = lcm.generate(
        concept_vec, max_concepts=max_concepts, temperature=temperature
    )
    return [sonar.decode([c]) for c in concept_plan]


def main() -> None:
    """Entry point for a small command-line demo."""

    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Why are renewables cheaper than fossil fuels?"
    )

    try:
        plan = generate_concept_plan(query)
    except SystemExit as exc:
        print(exc)
        return

    for i, sentence in enumerate(plan, 1):
        print(f"{i}. {sentence}")


if __name__ == "__main__":
    main()
