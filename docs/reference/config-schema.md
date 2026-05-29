# Config schema

Every config file declares its schema version at the top level with the `apiVersion` field, following the Kubernetes resource convention:

```yaml
apiVersion: promforecast.io/v1

datasource:
  url: http://victoriametrics:8428/

groups:
  - name: g
    queries:
      - id: my_metric
        promql: up
```

## Supported versions

| `apiVersion`                    | Status      | Notes |
|---------------------------------|-------------|-------|
| `promforecast.io/v1`            | Stable      | Frozen schema. Field additions are backward-compatible; removals require a major-version bump. |
| `promforecast.io/v1alpha1`      | Deprecated  | Byte-identical to `v1`. Loads with a `DeprecationWarning`; update at your next config edit. |

A future breaking change will ship under a new `apiVersion` (e.g. `promforecast.io/v2`). The forecaster will accept both versions during a documented deprecation window, and a `promforecast migrate` recipe will be provided.

## Loader behaviour

| Input                                    | Behaviour |
|------------------------------------------|-----------|
| Field omitted                            | Loads with default `promforecast.io/v1`, emits `DeprecationWarning` + log line `config_api_version_missing`. |
| `apiVersion: promforecast.io/v1`         | Loads silently. |
| `apiVersion: promforecast.io/v1alpha1`   | Loads with `DeprecationWarning` + log line `config_api_version_deprecated`. |
| Anything else                            | Refuses to start with `UnsupportedApiVersionError` listing supported versions. |

The version check runs **before** pydantic validation, so error messages stay precise (e.g. a typo produces a clear “unsupported apiVersion” message instead of a noisy Literal mismatch).

## Migrating from `v1alpha1` to `v1`

The `v1` schema is a verbatim freeze of `v1alpha1` — no field changes, no behaviour changes. Migration is one line:

```diff
-apiVersion: promforecast.io/v1alpha1
+apiVersion: promforecast.io/v1
```

No `helm upgrade` needed; the rolling reload picks up the change via the ConfigMap watcher. The forecaster continues to accept `v1alpha1` for at least one full minor release.

## Stability commitments (from `v1` onward)

- Top-level keys (`datasource`, `server`, `safety`, `defaults`, `sink`, `highAvailability`, `query_cache`, `telemetry`, `groups`, `triggers`, `annotations`) are part of the stable contract — no removals or renames.
- Field defaults may change in minor releases **only** when the new default is safer.
- Behavioural defaults that operators rely on (e.g. `defaults.accuracy.evaluate: true`) are pinned.
- New optional fields are additive and never bump the `apiVersion`.

## Why versioning matters

GitOps users commit configs to long-lived repositories. An explicit version lets us:
1. Ship a new schema under a fresh `apiVersion`.
2. Keep the old schema working during a deprecation window.
3. Provide a clear, one-shot migration path.

Without the label, every breaking change would force either dangerous silent coercion or a hard refusal to load. One line of YAML removes that dilemma.
