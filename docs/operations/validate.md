# `promforecast validate`

The validator is the first gate CI runs against every config PR. It fails fast on bad YAML, prints a full cardinality estimate, and can optionally run a cost projection or a live sample fit.

```bash
promforecast validate --config /path/to/config.yaml \
    [--strict-cardinality] \
    [--estimate-cost] \
    [--dry-run [--datasource-url URL]]
```

## Modes

| Flag                     | What it does |
|--------------------------|--------------|
| (default)                | Parse + schema validation + cardinality estimate |
| `--strict-cardinality`   | Same as default, but exits `1` if projected total exceeds `safety.max_total_series` |
| `--estimate-cost`        | Runs synthetic-data fits per model and prints projected CPU/memory budget per refresh |
| `--dry-run`              | Runs a real sample fit against the datasource and prints first 10 forecast points + MAPE preview |

Exit codes:  
- `0` — success  
- `1` — validation failure (schema, cardinality, etc.)  
- `2` — dry-run or cost-estimate failure (fit raised)

The command never modifies the config and only touches the datasource when `--dry-run` is used.

## Cardinality estimate

Printed as a per-family breakdown plus grand total. The estimate assumes every query returns `safety.max_series_per_query` series (worst case). Real deployments usually emit fewer lines, but the worst-case number is what can trip the global cap.

See [cardinality.md](cardinality.md) for the full multiplier table.

## Cost estimate (`--estimate-cost`)

Runs one synthetic fit per configured model (720-point hourly sinusoid + trend) and projects the wall-clock time and peak memory across the real config.

Output example:

```
Estimated runtime cost (synthetic-data fit per model):
  group node_capacity:
    refresh_interval               3600.0s
    estimated series                  1000
    per-fit AutoARIMA               0.142s
    per-fit SeasonalNaive           0.003s
    total fit time / refresh      145.00s
    peak memory (tracemalloc)    8421376 bytes

  total fit time across groups (per refresh): 145.00s
```

A `WARN:` line appears when a group’s projected total exceeds its `refresh_interval`. This is advisory only — the command still exits `0` unless `--strict-cardinality` is also used.

**Accuracy note**  
The estimator is order-of-magnitude only. Real fits on noisy data are typically 1–4× slower. Regressor fan-out and parallelism are not modelled. Treat large discrepancies (e.g. 100×) as real problems.

## Recommended CI shape

```yaml
- name: Validate promforecast config
  run: |
    promforecast validate \
      --config charts/promforecast/values.yaml \
      --strict-cardinality \
      --estimate-cost
```

This combination catches schema errors, oversized cardinality, and unexpectedly expensive model sets before they reach production.
