# Security Policy

## Supported releases

Only the latest minor release receives security fixes. An explicit support
matrix will be published once the project reaches a stable release.

| Release | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| older   | :x:                |

## Reporting a vulnerability

**Do not open a public issue for security problems.**

Please report vulnerabilities privately via [GitHub Security
Advisories](https://github.com/esops-dev/promforecast/security/advisories/new).

Include in the report:

- A description of the issue and its impact.
- Steps to reproduce, or a proof-of-concept.
- Affected versions, if known.
- Any suggested mitigation.

You should receive an acknowledgement within **5 business days**. We aim to
provide an initial assessment within **10 business days** and a fix or
mitigation timeline shortly after.

## Scope

In scope:

- The forecaster application (`forecaster/`).
- The Helm charts (`charts/promforecast`, `charts/promforecast-stack`).
- Container images published from this repository.
- Build, release, and supply-chain plumbing in `.github/`.

Out of scope:

- Vulnerabilities in upstream dependencies (please report to those projects;
  we will pick up the fix on the next dependency bump).
- Issues that require an attacker to already have cluster-admin or equivalent
  privileges in the target Kubernetes cluster.
- Denial-of-service caused by user-supplied configuration that exceeds the
  documented `safety:` limits — these are tuning parameters, not security
  boundaries.

## Disclosure

We follow coordinated disclosure. Once a fix is released we will publish an
advisory describing the issue, affected versions, and credit the reporter
unless they request otherwise.

## Supply chain

- Container images are built from `forecaster/Dockerfile` in CI; the build
  is reproducible from a tagged commit.
- Dependency manifests (`pyproject.toml`, chart `Chart.lock`) are pinned;
  Dependabot opens PRs for updates.
- Releases include an SBOM (Syft) and are scanned with Trivy in CI.
- We do not currently sign container images. Image signing with cosign is
  on the backlog.
