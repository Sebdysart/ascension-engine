# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Ascension Engine v2.1 — AI-powered short-form video production system built on Remotion + TypeScript + Python. No external services required; everything runs locally with file-based state (SQLite + JSON manifests).

### Services

| Service | Start command | Port | Notes |
|---------|--------------|------|-------|
| Remotion Studio | `npx remotion studio remotion/index.ts` | 3000 (default) | Main video composition preview |
| Ingest Pipeline | `python3 data/ingest.py` | N/A | Watch mode for video ingest; use `--dry-run --once <file>` for testing |
| Weekly Analyzer | `python3 data/analyze.py` | N/A | Reads `data/analytics.db`; use `--report-only` for safe testing |

### Gotchas

- The `remotion/` directory name collides with the `remotion` npm package. Do **not** set `baseUrl` in `tsconfig.json` to `"."` — TypeScript will resolve bare `"remotion"` imports to the local directory instead of `node_modules`.
- `moduleResolution` must be `"bundler"` in `tsconfig.json` for Remotion v4 type exports to resolve correctly.
- The `remotion/index.ts` entry point must call `registerRoot(RemotionRoot)` for the Studio to load compositions. Without it, Studio shows "Waiting for registerRoot()."
- Python pip installs go to `~/.local/bin`. Ensure `$HOME/.local/bin` is on `PATH` for the `scenedetect` CLI binary.
- The analytics SQLite DB must be initialized before `analyze.py` can run: `sqlite3 data/analytics.db < data/schema.sql`
- Standard dev commands are in `package.json` scripts: `npm run studio`, `npm run lint`, `npm run typecheck`.
- Remotion render example: `npx remotion render remotion/index.ts AscensionVideo out/video.mp4`
- Python deps are in `requirements.txt`; all optional (librosa, imagehash, watchdog) degrade gracefully if absent.
