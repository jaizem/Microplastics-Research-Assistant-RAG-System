# Evaluation

Default (same as before):

```powershell
python eval/run_eval.py
python main.py
```

Uses `data/eval/ragas_samples.json`.

Gold benchmark (same eval pipeline, questions from `data/benchmark/questions.jsonl`):

```powershell
python eval/run_eval.py --gold --limit 2
```

Writes `data/eval/gold_ragas_samples.json`, then runs the eval pipeline.

Reference-based metrics are included when `reference_answer` is present and `answerable_flag` is true:

- `answer_correctness`
- `context_precision`
- `context_recall`
