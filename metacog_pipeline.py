"""End-to-end planner→realiser demonstration."""

from __future__ import annotations

from lcm_pipeline import generate_concept_plan
from plan_realiser import Plan, Step, plan_to_text


def run_pipeline(query: str) -> str:
    """Generate a short answer to ``query`` using the LCM planner and realiser."""

    concept_sentences = generate_concept_plan(query)
    plan = Plan(
        dialogue_act="Inform",
        topic=query,
        steps=[Step(role="concept", text=s) for s in concept_sentences],
    )
    return plan_to_text(plan)


def main() -> None:
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Why are renewables cheaper than fossil fuels?"
    )
    try:
        result = run_pipeline(query)
    except SystemExit as exc:  # missing dependencies
        print(exc)
        return
    print(result)


if __name__ == "__main__":
    main()
