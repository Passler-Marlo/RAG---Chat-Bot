import json
from dataclasses import dataclass, asdict
from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class Step:
    role: str
    text: str


@dataclass
class Plan:
    dialogue_act: str
    topic: str
    steps: List[Step]


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
    print(plan_to_text(example_plan))
