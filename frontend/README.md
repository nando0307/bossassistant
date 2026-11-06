# BossAssistant frontend

React, TypeScript and Vite policy-question workspace for HR and Finance.

## Run locally

From `frontend/`:

```sh
npm ci
cp .env.example .env.local
npm run dev
```

`.env.example` sets `VITE_API_URL` to the local backend origin (`http://127.0.0.1:8000`, the same value the code falls back to). Point it at the deployed API URL from the root README to test against the hosted backend. This is a public build-time setting, not a place for secrets. The API must allow the frontend origin through CORS. Restart Vite after changing the environment.

```sh
npm run build
npm run lint
npm run preview
```

## Interaction and API contract

- Questions use the complete-answer `POST /ask` contract: `{ question, department, mode }`. Auto sends `department: null`; explicit choices send `hr` or `finance`. Search mode remains `fast` or `deep`. This UI does not use `/ask/stream`.
- Responses display `answer`, `department_routed`, and `sources`. Answer line breaks are preserved as plain text; backend text is not interpreted as HTML. Source titles, previews and identifiers stay beside their own answer, including earlier answers. Identifiers are not assumed to be public URLs. Empty source lists are explicitly identified.
- Pending requests lock the question and options so completion cannot erase a newly edited draft. Failed questions remain in the form. `Try again` submits the current question and options; it is never an automatic retry. No provider health or failover status is inferred.
- Demo identities come from `GET /auth/personas`; `POST /auth/demo` exchanges the selected identity for its token. Submission waits for discovery and the selected demo identity's token, including during access retries. Only a 404 confirms demo access is disabled; connection/server failures instead block submission and offer an access retry. A 401 requires renewed demo access or administrator assistance. This frontend does not add a production login or token-entry product.
- Answers are held only in the current page's memory and disappear on reload. No chat persistence is added.

Run the access-readiness regression check with Node 22.18+ from `frontend/`:

```sh
node --test tests/access.test.ts
```

## Accessibility and responsive behavior

The question comes first, with visible labels and a keyboard skip link. Department and depth buttons expose their pressed state; demo identity uses a native select. Examples use native disclosure. Controls have at least 44px target height, visible focus outlines and reduced-motion support. Pending and access messages use polite live regions; request errors use an alert. Successful completion focuses the latest answer heading. Sources stack immediately below their answer on narrow screens.

Manual browser smoke check:

1. Use Tab and Enter to follow **Skip to question**, enter a question, select HR/Finance and Fast/Deep, then submit. Confirm the request payload and the selected states.
2. While the API is pending, confirm the question/options are disabled and a waiting message appears, without claiming provider progress.
3. Return an answer containing paragraphs and citations. Confirm line breaks, source previews, and focus on the answer heading. Ask again and confirm each answer retains its own sources.
4. Return HTTP 503, then retry successfully. Confirm the question and choices survive the error. Return 401 and renew demo access before retrying. Keep persona lookup pending, fail it, and retry: submission must stay disabled throughout, until a selected-persona token arrives. Distinguish this from a 404-disabled deployment.
5. Return an empty source list and confirm the missing-evidence warning. Check 320px, 390px, 768px, 1024px and 1440px viewports for horizontal overflow and readable answers.

When live credentials/data are unavailable, use an isolated local API with synthetic content and the same contracts. This proves UI behavior only, not production authentication, retrieval quality or provider availability.

## Design source

Applied guidance from [UI/UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill/tree/b7d5353e0130feb8c5574ef1251f8a1db1873ccb), revision `b7d5353e0130feb8c5574ef1251f8a1db1873ccb`. Upstream tooling was used outside application source; no skill repository or new dependency is included.

Actual searches, with `UPSTREAM` pointing to that disposable checkout:

```sh
python3 "$UPSTREAM/src/ui-ux-pro-max/scripts/search.py" 'internal HR Finance knowledge assistant' --design-system -p 'Boss Assistant'
python3 "$UPSTREAM/src/ui-ux-pro-max/scripts/search.py" 'loading error recovery' --domain ux -n 3
python3 "$UPSTREAM/src/ui-ux-pro-max/scripts/search.py" 'form accessibility focus' --stack react -n 3
python3 "$UPSTREAM/src/ui-ux-pro-max/scripts/search.py" 'keyboard focus visible' --domain ux -n 3
python3 "$UPSTREAM/src/ui-ux-pro-max/scripts/search.py" 'financial trustworthy' --domain typography -n 1
```

Used the enterprise-app Minimalism & Swiss Style recommendation: functional grid, clear hierarchy, high contrast, navy `#0F172A`, secondary `#1E3A8A`, accessible gold action `#A16207`, background `#F8FAFC`, and the existing system-font stack. Typography makes no external font requests. Applied visible focus, error recovery, labeled controls and progressive disclosure guidance. The generator's marketing conversion sections and modal-specific advice do not fit this existing question workspace and were not applied.
