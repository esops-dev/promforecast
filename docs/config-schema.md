# Config schema

Every config file declares its schema version through the top-level `apiVersion` field, mirroring the Kubernetes resource convention:

```yaml
apiVersion: promforecast.io/v1alpha1

datasource:
  url: http://victoriametrics:8428/

groups:
  - name: g
    queries:
      - id: my_metric
        promql: up
```

## Currently supported versions

| `apiVersion`                    | Status  | Notes                                                            |
|---------------------------------|---------|------------------------------------------------------------------|
| `promforecast.io/v1alpha1`      | Current | The only accepted value while the schema is still alpha.         |

A stable schema identifier will be added when the contract is frozen (see `ROADMAP.md`). The forecaster will route between versions transparently and, when needed, ship a migration script alongside the breaking version.

## Loader behaviour

| Input                                    | Behaviour                                                                                                |
|------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Field omitted                            | Loads, defaults to `promforecast.io/v1alpha1`, emits a `DeprecationWarning` on stderr and a `config_api_version_missing` log line. |
| `apiVersion: promforecast.io/v1alpha1`   | Loads silently.                                                                                          |
| Anything else (typo, future version)     | Raises `UnsupportedApiVersionError` with the supported set; the forecaster refuses to start.             |

The check runs before pydantic validation so the operator-facing error message is precise — a typo like `promforecast.io/v1` produces `unsupported apiVersion 'promforecast.io/v1'; supported versions: promforecast.io/v1alpha1`, not the noisier `Literal[...]` mismatch pydantic would otherwise produce.

## Why versioning now

The schema is still alpha, but operators are starting to commit configs to long-lived GitOps repos. Pinning a version today means a future breaking change can:

1. Ship the new schema under a fresh `apiVersion`.
2. Keep the old schema accepted during a deprecation window.
3. Ship a one-shot migration tool (or recipe) that targets the old version unambiguously.

Without a version label, every breaking change would have to either silently coerce (dangerous) or refuse to load (operator-hostile). The label costs one line of YAML and removes that dilemma.
