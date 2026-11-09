# Fast-mode quality report: routing and abstention fixes

## Measured runs

| Run | Passed | Department routes | Source hits | Quality matches | HTTP errors |
| --- | --- | --- | --- | --- | --- |
| Frozen baseline `evals/baseline_fast.json` | 64/75 | 72/75 | 68/75 | 67/75 | n/a |
| Prior live run (semantic cache enabled) | 64/75 | 70/75 | 69/75 | 65/75 | not recorded |
| This branch, live rerun (semantic cache disabled) | 63/75 | 75/75 | 67/75 | 65/75 | 0 |

The three rows are not directly comparable: the baseline and the branch run used
a different model route and the prior live run had the semantic cache enabled,
which answers repeated questions without retrieval. The recorded numbers
therefore do **not** establish an aggregate improvement — the fully-passed count
moved from 64 to 63 while department routing moved from 72/75 (frozen baseline)
and 70/75 (prior live run) to 75/75. Raw result rows and credentials stay out of
the repository.

## What is evidenced

- Department routing: 75/75 correct in the recorded rerun, up from 70/75 live and
  72/75 in the frozen baseline. An offline replay of `route_question_fast` over
  all 75 eval questions agrees with every expected department, so
  `hr_ld_budget_wording`, `hr_tuition_reimbursement` and
  `both_conference_travel_approval` now reach the right index or both indexes.
- The intermittent HTTP 500s seen during live questions did not reproduce in this
  run (0 HTTP errors); the cause is still unexplained and is not claimed fixed.

## Still failing (retrieval coverage or generation, not routing)

The recorded run still loses quality matches relative to the frozen baseline, and
these baseline failures are untouched by any changed line:

- `fin_external_auditor` — `FIN-007` is never retrieved; no prompt change can
  supply support that retrieval did not return.
- `finance_international_per_diem`, `fin_domestic_meal_per_diem` — the right
  figure is answered but `FIN-001` loses the top-k slot to `FIN-002`/`FIN-010`.
- `both_new_hire_first_trip` — routing is correct; `FIN-001` is missing.
- `hr_tuition_clawback` — answers the clawback but omits the 12-month window.
- `hr_review_self_assessment` — the response arrived as repeated `<unk>` tokens,
  i.e. a corrupted generation from the gateway, not a policy or routing defect.

## Reproducing the comparison

Run the unchanged suite against a freshly written result file, then gate it:

```
uv run python scripts/run_eval.py --api-url http://127.0.0.1:8000 --token "$TOKEN" --output evals/results_fast.jsonl
uv run python scripts/eval_gate.py check evals/results_fast.jsonl --baseline evals/baseline_fast.json
```

Diff the per-case `passed`/`abstained`/`department_actual` fields against the
failing-case list above. Note that `scripts/run_eval.py` defaults to
`evals/results.jsonl`; without `--output` the gate compares the frozen baseline
against itself and passes trivially.
