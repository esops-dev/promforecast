# Verifying a reload picked up the expected changes

The forecaster reloads its config on three triggers:

- `POST /-/reload`
- `SIGHUP`
- ConfigMap watcher (`server.reload.watch_configmap: true`, default)

All three paths use the same validate-then-swap logic. A successful reload always emits:

1. One `config_reload_succeeded` line.
2. One `group_queries_loaded` line **per group**, listing the active query IDs.

```json
{"event":"config_reload_succeeded","added":[],"removed":[],"changed":["node_capacity"],"config_hash":"..."}
{"event":"group_queries_loaded","group":"node_capacity","queries":["node_cpu_busy_pct","node_memory_used_bytes","node_load1","node_load5"],"source":"reload"}
```

**Recommended verification method:** Check the `group_queries_loaded` lines after every reload. They show exactly what the forecaster is running.

## Why this matters: the partial-config trap

`config_reload_succeeded` only confirms the YAML parsed correctly.  
On some platforms (notably macOS Docker Desktop bind-mounts), a fast edit can trigger a reload on a **truncated** file. The forecaster accepts the partial config and runs with missing queries.

The `group_queries_loaded` lines are the only reliable way to detect this. A missing query ID is the smoking gun.

## Log line fields

| Field     | Meaning |
|-----------|---------|
| `event`   | Always `group_queries_loaded` |
| `group`   | Group `name` (one line per group, in YAML order) |
| `queries` | List of query `id`s for that group (in YAML order) |
| `source`  | `boot` or `reload` |

Empty groups emit `queries: []` so missing queries are never ambiguous.

## Boot-time emission

The same lines appear at startup (with `source: boot`), right after `config_loaded`:

```json
{"event":"config_loaded","path":"/etc/promforecast/config.yaml","groups":4}
{"event":"group_queries_loaded","group":"node_capacity","queries":["..."],"source":"boot"}
```

## Recovery if truncation occurred

1. Confirm by inspecting the latest `group_queries_loaded` lines — they always reflect reality.
2. Force a clean reload: make a trivial change (e.g. add/remove a comment) to update the file hash, then trigger a reload.
3. If the problem persists, restart the pod to get a fresh mount.

There is no separate “config status” endpoint by design — the `group_queries_loaded` logs are the source of truth.
