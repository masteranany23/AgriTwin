# AgriTwin

AgriTwin is a physics-based agricultural digital-twin platform that combines WOFOST/PCSE crop simulation, remote sensing observations, weather and soil data, and EnKF assimilation.

## Run & Operate

- `pnpm --filter @workspace/agritwin-web run dev` — run the AgriTwin web artifact through its managed workflow
- `python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000` — run the preserved Python backend locally after installing `requirements.txt`
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- Web: React + Vite, Tailwind CSS, Recharts
- Existing scientific backend: Python, FastAPI, SQLAlchemy, PCSE/WOFOST
- Existing database layer: PostgreSQL/SQLite configuration with Alembic migrations

## Where things live

- `backend/app/` — preserved FastAPI application and scientific modules
- `alembic/` — preserved database migrations
- `tests/` — preserved backend test suite
- `artifacts/agritwin-web/` — React web artifact
- `artifacts/api-server/` — workspace API scaffold; the Python backend remains the source of truth for AgriTwin scientific behavior
- `attached_assets/` — uploaded design brief and reference assets

## Architecture decisions

- The existing Python backend is frozen and remains the source of truth for simulation and assimilation behavior.
- The AgriTwin web artifact is separate from the backend so the frontend can evolve without changing scientific code.
- The v0.1 web experience uses deterministic local demonstration data until a later task wires it to the backend APIs.

## Product

The product demonstrates crop simulation, field observations, state assimilation, and forecast trajectories for agricultural researchers and institutional stakeholders.

## User preferences

- Keep the existing backend scientific implementation and tests unchanged.
- Prefer small, focused changes over restructuring the scientific stack.

## Gotchas

- The original backend uses Python dependencies from `requirements.txt`; the generated Node workspace API scaffold is not a replacement for it.
- Do not run `pnpm dev` at the workspace root; use the configured artifact workflows.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
