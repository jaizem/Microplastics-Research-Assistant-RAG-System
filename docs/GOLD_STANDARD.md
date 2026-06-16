# Gold benchmark

`data/benchmark/questions.jsonl` — one JSON object per line. Ground-truth Q&A for eval, not model output.

Each row:

- `question_id` — e.g. mp-001
- `question`
- `reference_answer` — short answer from the papers; scored via answer correctness, context precision, and context recall
- `source_chunk_ids` — optional chunk labels for human review, e.g. `microplastic_health_effects__33`
- `paper_id` — PDF stem in `data/context_corpus/`, or null
- `answerable_flag` — false if the corpus cannot answer (reference-based metrics are skipped for these rows)
- `difficulty` — easy, medium, or hard

To add a row: copy a line from `data/benchmark/ignored-question.template.jsonl`, set a new `question_id`, fill in answers from the PDF.

Run: `python eval/run_eval.py --gold --limit 2`
