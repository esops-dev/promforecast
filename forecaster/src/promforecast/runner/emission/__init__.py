"""Per-family emission helpers for the runner.

Each module here owns the pure-function pieces of one emission family —
the parts that can be reasoned about without a live :class:`~promforecast.runner.Runner`
instance. Methods that need instance state (an executor, a config, an
exporter) stay on ``Runner`` itself in ``scheduler``; this package
collects the rest so a contributor extending an emission family touches
one focused file instead of scrolling through the orchestration layer.
"""
