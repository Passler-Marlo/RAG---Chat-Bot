try:
    from lcm import LCM
    from sonar import SonarModel
except ImportError as e:
    raise SystemExit("LCM and SonarModel libraries are required: {}".format(e))


def generate_concept_plan(query: str, max_concepts: int = 4, temperature: float = 0.4):
    """Generate a list of concept sentences for the given query."""
    # Load pretrained models
    lcm = LCM.from_pretrained("base_lcm")
    sonar = SonarModel.from_pretrained("sonar_all")

    # Encode the query into concept vector
    concept_vec = sonar.encode([query])

    # Generate sequence of concept vectors
    concept_plan = lcm.generate(concept_vec, max_concepts=max_concepts, temperature=temperature)

    # Decode each concept vector back to text for inspection
    return [sonar.decode([c]) for c in concept_plan]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Why are renewables cheaper than fossil fuels?"

    plan = generate_concept_plan(query)
    for i, sentence in enumerate(plan, 1):
        print(f"{i}. {sentence}")
