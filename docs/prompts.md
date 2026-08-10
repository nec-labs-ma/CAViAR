# Prompt templates

All models receive identical task prompts (aside from chat templates).

## Multiple-choice

```
{context}

{question}
A. {choice_0}
B. {choice_1}
...
Answer with ONLY the letter (e.g., A, B, C) of the correct option.
```

Context strings open with: *“You are analyzing a dashcam video of a traffic incident,”* followed by a short task cue (attend to weather/road surface, or observe vehicle interaction for accident type).

Options use a **fixed canonical order** (not randomized per item).

## Open-ended (responsibility)

```
{context}
{question}
{instruction}
```

Examples of instructions:

- **At-fault:** identify the vehicle at fault and briefly explain why  
- **Affected:** identify the victim/affected vehicle and explain why  
- **Violation:** specify the traffic rule that was violated and which vehicle violated it  

## LLM-as-Judge (GPT-4o)

**System:**

```
You are an expert evaluator for traffic accident analysis.
You will compare a model's prediction against a ground truth answer
and rate how well the prediction matches.
```

**User:**

```
Given the following question about a traffic accident video:

Question: {question}

Ground truth answer: {gt}

Model prediction: {pred}

Rate the model's prediction on a scale of 0 to 5:
  5 = The prediction perfectly matches the ground truth in meaning,
      identifying the same vehicle(s) and the same reasoning.
  4 = The prediction almost completely matches — minor wording
      differences but all key facts are correct.
  3 = The prediction mostly matches — identifies the correct
      vehicle(s) but misses some reasoning details or includes
      minor inaccuracies.
  2 = The prediction partially matches — captures some correct
      aspects but misses key details or gets some elements wrong.
  1 = The prediction slightly matches — only a small part is
      relevant or correct, most is wrong or missing.
  0 = The prediction does not match the ground truth at all,
      or is irrelevant.

Respond with ONLY a single line in this exact format:
Score: <number>
```

Temperature = 0. No few-shot examples.
