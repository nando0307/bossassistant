# Fast-mode quality report: routing and abstention fixes

Baseline: `evals/results_fast.jsonl` + `evals/baseline_fast.json` (75 factoid cases,
75 ok, 64 passed, department_match 72, source_hit 68, quality_match 67). The
baseline files are frozen harness artifacts and are unchanged by this change, so
the numbers below are the *before* half of the comparison only.

## What the change targets

| Failed case | Observed failure | Addressed by |
| --- | --- | --- |
| `hr_ld_budget_wording` | routed `finance`, abstained | HR benefit terms win over generic `budget`/`reimbursement` in `route_question_fast` |
| `hr_tuition_reimbursement` | routed `finance`, abstained | same |
| `both_conference_travel_approval` | routed `finance`, HR half dropped | `conference` in `HR_TERMS` |
| `adversarial_pto_inflation` | abstained on a false premise | claim-confirmation instruction in `RAG_TEMPLATE` |
| `fin_expense_deadline` | abstained although `FIN-001` was retrieved | removed the "general reimbursement rules" abstention rule |

## Not addressed here (retrieval coverage, not generation)

- `fin_external_auditor` — `FIN-007` is never retrieved; no prompt change can
  supply support that retrieval did not return.
- `finance_international_per_diem`, `fin_domestic_meal_per_diem` — the right
  figure is answered but `FIN-001` loses the top-k slot to `FIN-002`/`FIN-010`.
- `both_new_hire_first_trip` — routing is already correct; `FIN-001` is missing.
- `hr_tuition_clawback` — answers the clawback but omits the 12-month window.
- `hr_review_self_assessment` — the response arrived as repeated `<unk>` tokens,
  i.e. a corrupted generation from the gateway, not a policy or routing defect.

## Comparison limitations

No post-change result file is committed: producing one needs a live API, Neo4j
and gateway credentials. To reproduce the comparison, run the unchanged suite

```
uv run python scripts/run_eval.py --api-url http://127.0.0.1:8000 --token "$TOKEN"
uv run python scripts/eval_gate.py check evals/results_fast.jsonl --baseline evals/baseline_fast.json
```

and diff the per-case `passed`/`abstained`/`department_actual` fields against the
table above. Until that run exists, this document records intended effect only.
