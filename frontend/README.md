# DeepSecure MVP (frontend)

This is a small TypeScript + React (Vite) frontend that implements a premium "DeepSecure" results UI with a neon, glass aesthetic. It includes a mocked `detectDeepfake` function and hooks to call a backend at `http://localhost:8000/detect`.

Quick run (frontend):

1. cd frontend
2. npm install
3. npm run dev

Backend:
- The repo contains a FastAPI backend at `backend/`. Run it with:

  python -m uvicorn backend.main:app --reload --port 8000

Integration:
- The frontend's `src/components/mockApi.ts` will try to POST to `http://localhost:8000/detect` if you set `preferBackend` to `true`.

What maps where:
- `DeepfakeResult` fields are mapped exactly to the premium Result panel and to the exact formatted copy block in `ResultPanel.tsx`.
- The ISO timestamp string from the response is displayed verbatim under "Timestamp:" and a local formatted time is shown below it.

Tests:
- Basic tests (formatting, copy) are included; run `npm run test` after installing deps.

Where to plug a real API:
- Replace the mock `detectDeepfake` in `src/components/mockApi.ts` or set `preferBackend = true` to attempt a POST to `http://localhost:8000/detect` and return parsed JSON.

Notes:
- This is a demo/mvp; styling is done with Tailwind and a fallback CSS file. You can easily extend it.
