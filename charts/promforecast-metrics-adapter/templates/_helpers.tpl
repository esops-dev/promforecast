{{/*
Common helpers for the promforecast-metrics-adapter chart.
*/}}

{{- define "promforecast-metrics-adapter.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "promforecast-metrics-adapter.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "promforecast-metrics-adapter.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "promforecast-metrics-adapter.labels" -}}
helm.sh/chart: {{ include "promforecast-metrics-adapter.chart" . }}
{{ include "promforecast-metrics-adapter.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: promforecast
app.kubernetes.io/component: metrics-adapter
{{- end -}}

{{- define "promforecast-metrics-adapter.selectorLabels" -}}
app.kubernetes.io/name: {{ include "promforecast-metrics-adapter.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "promforecast-metrics-adapter.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "promforecast-metrics-adapter.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "promforecast-metrics-adapter.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}

{{/*
TLS certificate (self-signed when tls.generate=true).

Stored in a ``hiddenSecret`` flavour so re-rendering on each
``helm upgrade`` regenerates the cert. Stable across upgrades within
the same release because the chart pins the CN to the service DNS
name; only the validity period rolls.
*/}}
{{- define "promforecast-metrics-adapter.tlsSecretName" -}}
{{- if .Values.tls.existingSecret -}}
{{- .Values.tls.existingSecret -}}
{{- else -}}
{{- printf "%s-tls" (include "promforecast-metrics-adapter.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
TLS material generation.

``genCA`` / ``genSignedCert`` are non-deterministic across calls,
so the cert is generated *once* per Helm render and stashed on
``$.Values`` for the duration of that render. Subsequent calls
(e.g. one for the Secret, one for the APIService) pull the
already-generated material instead of producing a fresh CA.

On ``helm upgrade`` against a live cluster, ``lookup`` finds the
chart-managed Secret from the previous install and the helper
returns *that* material — so the cert + CA bundle stays stable
across upgrades and the APIService caBundle doesn't churn (which
would otherwise force a rolling restart on every upgrade via the
``checksum/tls`` annotation and break in-flight HPA polls). The
fresh-generate path only fires when ``lookup`` returns nothing:
first install, or offline ``helm template`` (which is fine — the
golden-file render is deterministic per process but irrelevant to
a real cluster).

Returns a dict with ``ca``, ``cert``, ``key`` (PEM strings). For
``tls.existingSecret`` installs the helper uses ``lookup`` against
the operator-named Secret and returns its data instead.
*/}}
{{- define "promforecast-metrics-adapter.tlsMaterial" -}}
{{- $fullname := include "promforecast-metrics-adapter.fullname" . -}}
{{- if .Values.tls.existingSecret -}}
{{- $secret := lookup "v1" "Secret" .Release.Namespace .Values.tls.existingSecret -}}
{{- if $secret -}}
{{- dict "ca" (index $secret.data "ca.crt" | b64dec) "cert" (index $secret.data "tls.crt" | b64dec) "key" (index $secret.data "tls.key" | b64dec) | toYaml -}}
{{- else -}}
{{- /*
The operator set ``tls.existingSecret`` but the Secret isn't in the
cluster (or we're rendering offline via ``helm template`` which
returns nothing from ``lookup``). Failing loud here is the right
default for a production install — the APIService would otherwise
fall back to ``insecureSkipTLSVerify: true`` and silently accept any
upstream cert, which defeats the existingSecret contract.

For offline rendering (CI's ``helm template`` golden-file path),
either pass ``--set tls.generate=true`` to exercise the self-signed
path, or pre-create the Secret in a kind cluster and let the lookup
populate.
*/}}
{{- fail (printf "tls.existingSecret=%q is set but Secret %q was not found in namespace %q. Either create the Secret first (it must carry ca.crt/tls.crt/tls.key), or unset existingSecret and use tls.generate=true. Note: ``helm template`` cannot resolve ``lookup`` against a live cluster; pass --set tls.generate=true to render offline." .Values.tls.existingSecret .Values.tls.existingSecret .Release.Namespace) -}}
{{- end -}}
{{- else -}}
{{- if not (hasKey .Values "_tlsMaterial") -}}
{{- $existing := lookup "v1" "Secret" .Release.Namespace (printf "%s-tls" $fullname) -}}
{{- if and $existing (index $existing.data "ca.crt") (index $existing.data "tls.crt") (index $existing.data "tls.key") -}}
{{- $_ := set .Values "_tlsMaterial" (dict "ca" (index $existing.data "ca.crt" | b64dec) "cert" (index $existing.data "tls.crt" | b64dec) "key" (index $existing.data "tls.key" | b64dec)) -}}
{{- else -}}
{{- $svcDNS := printf "%s.%s.svc" $fullname .Release.Namespace -}}
{{- $svcDNSFull := printf "%s.%s.svc.cluster.local" $fullname .Release.Namespace -}}
{{- $altNames := list $svcDNS $svcDNSFull $fullname -}}
{{- $ca := genCA (printf "%s-ca" $fullname) (int .Values.tls.validityDays) -}}
{{- $cert := genSignedCert $svcDNS nil $altNames (int .Values.tls.validityDays) $ca -}}
{{- $_ := set .Values "_tlsMaterial" (dict "ca" $ca.Cert "cert" $cert.Cert "key" $cert.Key) -}}
{{- end -}}
{{- end -}}
{{- .Values._tlsMaterial | toYaml -}}
{{- end -}}
{{- end -}}
