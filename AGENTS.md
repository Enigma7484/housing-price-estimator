# Codex App Factory Instructions

This repository should be treated as a reusable AI business-app factory, not only as a price-estimator demo.

## Product Direction

- Default product frame: vertical SaaS decision support.
- Current product: ResaleIQ, a trade-in and resale quote copilot.
- Prioritize buyer workflows over generic model showcases.
- Turn model outputs into operating decisions: quote, offer, approval, margin, risk, explanation, and history.
- Keep internal strategy notes out of the visible app unless the user explicitly asks for copy or documentation.

## Implementation Pattern

When creating another app from this codebase:

1. Pick one narrow buyer and one painful workflow.
2. Make the first screen the usable workflow, not a marketing page.
3. Keep FastAPI estimators and registry as the backend foundation.
4. Add a frontend translation layer that converts predictions into business decisions.
5. Add owner/operator visibility: saved history, margin/risk metrics, and review tables.
6. Keep old estimators available only when they support the new product story.

## Frontend Expectations

- Use React, TypeScript, Vite, Tailwind, and lucide-react patterns already in the app.
- Prefer dense operational screens: forms, tables, metrics, status chips, and action buttons.
- Avoid decorative landing-page work unless specifically requested.
- Preserve responsive behavior and verify mobile width for no horizontal overflow.

## Verification

Run these checks after meaningful frontend work:

```bash
cd frontend
npm run build
```

For full local verification:

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

```bash
cd frontend
npm run dev
```

Use `http://localhost:5173`, not `http://127.0.0.1:5173`, because the backend CORS config allows localhost for local development.

## Deployment

- The GitHub remote is the deployment source.
- Pushing `main` should trigger Vercel automatic deployment when the Vercel project is connected to this repository.
- Root `vercel.json` is the intended Vercel entrypoint for combined frontend/backend deployment.
- Do not change Vercel settings unless a deploy failure proves the config is wrong.

