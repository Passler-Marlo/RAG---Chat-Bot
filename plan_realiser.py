11gdeo-codex/implement-meta-s-large-concept-models-as-planner
"""Simple discourse plan realiser.

This module demonstrates how a small sequence-to-sequence model can be used
to realise an explicit discourse plan. It intentionally mirrors classic
surface realisation pipelines, with the plan supplied as JSON.  The code is
kept minimal so the focus remains on the metacognitive split between planning
and verbalisation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

__all__ = [
    "Step",
    "Plan",
    "plan_to_text",
]


def _load_model(model_name: str, *, local_files_only: bool = False) -> Tuple["AutoTokenizer", "AutoModelForSeq2SeqLM"]:
    """Return tokenizer and model for ``model_name``.

    Parameters
    ----------
    model_name:
        Name of the Hugging Face model to load.
    local_files_only:
        If ``True``, do not attempt to download weights.
    """

    try:  # Lazy import so the module can be used without transformers installed
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "The 'transformers' package is required: {}".format(exc)
        ) from exc

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=local_files_only)
    return tok, mod

import json
from dataclasses import dataclass, asdict
from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



@dataclass
class Step:
11gdeo-codex/implement-meta-s-large-concept-models-as-planner
    """A single clause in the discourse plan."""

    role: str
    text: str


@dataclass
class Plan:
11gdeo-codex/implement-meta-s-large-concept-models-as-planner
    """Complete discourse plan to be realised."""


    dialogue_act: str
    topic: str
    steps: List[Step]


11gdeo-codex/implement-meta-s-large-concept-models-as-planner
def plan_to_text(
    plan: Plan,
    model_name: str = "google/flan-t5-small",
    *,
    local_files_only: bool = False,
) -> str:
    """Realise ``plan`` into natural language using ``model_name``.

    Parameters
    ----------
    plan:
        Structured discourse plan.
    model_name:
        Name of the Hugging Face model to use.
    local_files_only:
        If ``True``, do not attempt to download weights. Useful in offline
        environments.
    """

    tokenizer, model = _load_model(model_name, local_files_only=local_files_only)

def plan_to_text(plan: Plan, model_name: str = "google/flan-t5-small") -> str:
    """Convert a discourse plan into fluent text using a small seq2seq model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    # JSON dump to maintain a stable format
    plan_json = json.dumps(asdict(plan), ensure_ascii=False)
    prompt = f"Realise the following plan into a short paragraph:\n{plan_json}"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=120)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

11gdeo-codex/implement-meta-s-large-concept-models-as-planner
def _default_plan() -> Plan:
    """Return the example plan used in the README."""
    return Plan(

if __name__ == "__main__":
    example_plan = Plan(

        dialogue_act="Inform",
        topic="Madonna",
        steps=[
            Step(role="claim", text="Madonna is an American singer and actress."),
            Step(role="evidence", text="She is often called the 'Queen of Pop'."),
            Step(role="evidence", text="She has won 7 Grammy Awards."),
            Step(role="wrap_up", text="Her influence on pop culture is widely recognised."),
        ],
    )
11gdeo-codex/implement-meta-s-large-concept-models-as-planner


def main() -> None:
    """Run a small demo using the default plan."""
    try:
        text = plan_to_text(_default_plan())
    except Exception as exc:  # pragma: no cover - network errors etc.
        print("Model could not be loaded:", exc)
        return
    print(text)


if __name__ == "__main__":
    main()
    print(plan_to_text(example_plan))
