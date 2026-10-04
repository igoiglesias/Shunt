# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-02

### Added
- **Relay catch-all** (`app/routers/relay.py`): unknown paths forwarded verbatim to official host with client credentials; Shunt token stripped; gzip bytes passed through un-decompressed via `aiter_raw()`. Relay events excluded from panel totals but recorded in audit logs.
- **Kind column** in `RequestEvent` model (`app/stats/models.py:58-66`): distinguishes model calls (`kind=model`) from relay passthroughs (`kind=relay`).
- **Panel aggregates exclude relay** (`app/stats/queries.py`): relay rows no longer pollute model latency/token charts.
- **Audit search filters by kind** (`app/stats/queries.py`, `app/routers/audit.py`): `kind` parameter in `search_events`, column in CSV export, "missing ≠ zero" semantics.
- **Audit UI shows relay rows** (`app/templates/audit.html`, `app/routers/audit.py`): filter, CSV, and "missing not zero" behavior implemented.
- **E2E tests for relay** (`tests/e2e/test_relay_e2e.py`): 7 scenarios covering catch-all, gzip, 404/405, stream, tools, fallback.
- **README sections** for "Routes Shunt does not implement" (relay) and updated relay docs.
- **OfficialHost.origin** (`app/core/official_hosts.py`): base URL without path/query/fragment for transparent bypass.
- **Structured observability** (`app/core/observability.py`): logging, stats Recorder, config watcher.

### Changed
- **--error token darkened** from `#d5463a` to `#c13a2e` (`app/templates/shunt.css:38`): white-on-error buttons now 5.37:1 (AA). `--error-text: #e2695c` kept for text-on-dark (5.67:1).
- **--dim token lightened** from `#8a8478` to `#a09a8d` (`app/templates/shunt.css:27`): 5.68:1 on `--panel-2`, 6.15:1 on `--panel` (both AA).
- **Sticky actions column on mobile** (`app/templates/admin_config.html:89-93`): `:last-child` selector reaches the real cell; opaque `--panel` background blocks scrolled content.
- **Touch targets 44px** (`app/templates/admin_config.html:96-98`): `.btn`/`.btn-sm` min-height in 760px media query.
- **Chips are labels, not controls** (`app/templates/_providers.html:20`, `_models.html:26-29`): `aria-pressed` removed from `<span class="chip">` (28 axe-core critical violations fixed).
- **chip.bad paints red** (`app/templates/shunt.css:229`): selector without dead state filter; real interactive `.chip.bad` in audit gets explicit pressed rule with light text.

### Fixed
- **Duplicate Set-Cookie collapse** (`app/routers/relay.py:98-122`): switched from `headers={}` dict to `raw_headers` assigned post-construction using `httpx.Headers.multi_items()`; preserves separate `Set-Cookie` headers with commas in `Expires`/`Date`.
- **Login error contrast regression** (`app/templates/login.html:19`): `.error` now uses `--error-text` (5.67:1) instead of darkened `--error` (3.46:1).
- **Dashboard .tape text contrast regression** (`app/templates/dashboard.html:203`): `.what b` now uses `--error-text` (5.29:1) instead of darkened `--error` (3.23:1).
- **Dashboard .tape tint stale** (`app/templates/dashboard.html:184`): hardcoded `rgba(213,70,58,0.10)` replaced with `rgba(var(--error-rgb), 0.10)` where `--error-rgb: 193, 58, 46` tracks `--error`.
- **Audit chip.bad pressed state** (`app/templates/shunt.css:235-239`): new rule with `color: var(--ink)` on `--error` background (4.67:1).
- **/v1 prefix precision** (`app/routers/relay.py:85`): `startswith("/v1")` → exact `/v1` or `/v1/` to avoid matching `/v1x`.
- **Type annotations and docstring fixes** across tests and relay code.

### Security
- No credential leaks in diff (verified by review gate).
- No bidirectional/homoglyph/zero-width characters.
- No supply-chain pins added (dependency upgrades fixed in code, not pinned).