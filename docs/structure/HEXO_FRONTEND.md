# HEXO_FRONTEND

## Purpose

`hexo-frontend` owns browser-facing local tools for Hexo. It can serve static
assets, local dashboards, and small development-only web servers.

The frontend may call stable engine, runner, or training APIs when those APIs
exist, but those packages must not import frontend modules.

## Owns

- Browser UI.
- Static assets.
- Local HTTP servers for development tools.
- UI-specific adapters that convert `PythonHexoState` into browser-facing
  dashboard state.
- Tactics overlay presentation, filtering, labels, summaries, and derived facts
  such as immediate wins or must-block placements.

## Does Not Own

- Game rule authority.
- Runner player contracts.
- Model architecture or tensors.
- Training orchestration.
- Durable run records.

## Package Layout

```text
packages/hexo_frontend/
  pyproject.toml
  python/
    hexo_frontend/
      __init__.py
      dashboard.py
      static/
        app.js
        index.html
        styles.css
      web.py
      py.typed
```

## Current Status

The package currently serves a simple manual-play Hexo board. The server creates
an interactive match through `hexo_runner.modes.match`, so the frontend does not
own game state or placement legality. The runner passes cloned `HexoState`
objects, and the frontend calls `hexo_engine.to_python_state()` when it needs a
read-only state mirror for rendering.

`dashboard.py` is the frontend adapter that turns `PythonHexoState` into the
JSON shape the browser needs. The Python server owns HTTP routing, API/static
asset serving, and frontend-only adaptation; board rendering, tactics overlays,
and inspector interaction live in package static assets.
