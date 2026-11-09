# Fast-mode quality report: routing and abstention fixes

## Measured runs

| Run | Passed | Department routes | Source hits | Quality matches | HTTP errors |
| --- | --- | --- | --- | --- | --- |
| Frozen baseline `evals/baseline_fast.json` | 64/75 | 72/75 | 68/75 | 67/75 | n/a |
| Prior live run (semantic cache enabled) | 64/75 | 70/75 | 69/75 | 65/75 | not recorded |
| This branch, live rerun (semantic cache disabled) | 63/75 | 75/75 | 67/75 | 65/75 | 0 |

The rows come from different sources — the frozen baseline in `evals/` was
measured through a different model route, and the prior live run had the semantic
cache enabled, which answers repeated questions without retrieval. The recorded
numbers therefore do **not** establish an aggregate improvement, and the two
64/75 totals do not mean the same thing: the fully-passed count moved from 64 to
63 while department routing moved from 72/75 (frozen baseline) and 70/75 (prior
live run) to 75/75. Raw result rows and credentials stay out of the repository.

## Recovered vs newly failing

The frozen baseline's 11 failures, per the `passed: false` cases in
`evals/baseline_fast.json`, are `finance_international_per_diem`,
`hr_ld_budget_wording`, `hr_tuition_reimbursement`, `hr_tuition_clawback`,
`fin_expense_deadline`, `fin_domestic_meal_per_diem`,
`fin_card_reconciliation_due`, `fin_external_auditor`,
`both_conference_travel_approval`, `both_new_hire_first_trip` and
`both_home_office_stipend_claim`.

The recorded rerun passes 63/75, i.e. 12 failing cases. Its per-case rows are not
committed, so which baseline failures recovered and which case newly failed
cannot be read from this repository, and because the model route and cache setting
differ from the baseline, the 64 → 63 movement is not attributed to this change
here. Reproducing the run with the commands below and diffing per-case `passed`
is what closes that gap.

The tracked raw run `evals/results_fast.jsonl` is not a substitute: it disagrees
with the frozen baseline on four cases (`hr_review_self_assessment` and
`adversarial_pto_inflation` pass in the baseline and fail in the raw file;
`fin_card_reconciliation_due` and `both_home_office_stipend_claim` the reverse),
so the two repository artifacts are not interchangeable. In that raw file the
`hr_review_self_assessment` failure is a generation corrupted into repeated
`<unk>` tokens, not a policy or routing defect.

## What is evidenced

- Department routing: 75/75 correct in the recorded rerun. Offline, the fast
  router decides 59 of the 75 questions and matches the expected department on
  every one of them; the other 16 (including `vague_how_much`, `hr_review_pip`,
  `fin_fiscal_year_end`, `oos_severance` and `vague_deadline`) carry no term the
  fast router recognizes and defer to the model router
  (`src/app/agents/router.py:226`). The 75/75 is the final observed count across
  both routers, not deterministic route coverage.
- The frozen baseline's three department mismatches — `hr_ld_budget_wording`,
  `hr_tuition_reimbursement` (routed `finance`) and
  `both_conference_travel_approval` (routed `finance`, HR half dropped) — are
  gone in the recorded rerun.
- The mixed benefit/budget routing limitation is not fixed: a question whose only
  finance signal is `budget` or `reimbursement` still routes HR alone.
- The intermittent HTTP 500s seen during live questions did not reproduce in this
  run (0 HTTP errors); the cause is still unexplained and is not claimed fixed.

## Remaining baseline failures with identified causes

- `fin_external_auditor` — `FIN-007` is never retrieved; no prompt change can
  supply support that retrieval did not return.
- `finance_international_per_diem`, `fin_domestic_meal_per_diem` — the right
  figure is answered but `FIN-001` loses the top-k slot to `FIN-002`/`FIN-010`.
- `both_new_hire_first_trip` — routing is correct; `FIN-001` is missing.
- `hr_tuition_clawback` — answers the clawback but omits the 12-month window.
- `fin_expense_deadline` — abstained although `FIN-001` was retrieved; targeted by
  the abstention prompt change, but whether it now passes is not evidenced by the
  aggregates above.
- `fin_card_reconciliation_due`, `both_home_office_stipend_claim` — quality match
  false in the frozen baseline, untouched by any changed line; the repository
  does not record why they failed there.

## Reproducing the comparison

Run the unchanged suite against a fresh result file, then gate it:

```
uv run python scripts/run_eval.py --api-url http://127.0.0.1:8000 --token "$TOKEN"
uv run python scripts/eval_gate.py check evals/results.jsonl --baseline evals/baseline_fast.json
```

`evals/results.jsonl` is the harness default and is gitignored, so a fresh run
never overwrites the tracked `evals/results_fast.jsonl` that the frozen baseline
was derived from. Diff the per-case `passed`/`abstained`/`department_actual`
fields against the failing-case list above.
