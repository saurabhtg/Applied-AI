# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
common/tracing.py

OpenTelemetry spans for every LLM call, retrieval, agent step, and eval.

The principle: if you can't see what your system did, you can't debug it.
Tracing is not a nice-to-have for LLM systems — it's the only way to
reconstruct what happened when something goes sideways at 3 a.m.

This module gracefully no-ops if OTel isn't configured, so your tests
and local scripts don't need a collector running.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, Status, StatusCode

    _OTEL_AVAILABLE = True
    _tracer = trace.get_tracer("applied-ai", "1.0.0")
except ImportError:
    _OTEL_AVAILABLE = False
    _tracer = None  # type: ignore


class _NoopSpan:
    """A minimal span shim used when OpenTelemetry isn't installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """
    Start an OpenTelemetry span.

    If OTel isn't installed or configured, yields a no-op span object that
    silently accepts attribute calls. Calling code doesn't need to change.

    Args:
        name: Span name (e.g. "llm.complete", "rag.retrieve", "agent.step").
        attributes: Optional attributes to set on the span at start.

    Yields:
        The active span object.

    Example:
        with span("rag.retrieve", attributes={"query": q}) as s:
            results = retriever.search(q)
            s.set_attribute("results.count", len(results))
    """
    if not _OTEL_AVAILABLE or _tracer is None:
        yield _NoopSpan()
        return

    with _tracer.start_as_current_span(name) as otel_span:
        if attributes:
            for key, value in attributes.items():
                # OTel attributes must be primitives — coerce safely.
                otel_span.set_attribute(key, _coerce_attribute(value))
        try:
            yield otel_span
        except Exception as exc:
            otel_span.record_exception(exc)
            otel_span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def _coerce_attribute(value: Any) -> Any:
    """Coerce arbitrary values to OTel-acceptable primitives."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_attribute(v) for v in value]
    return str(value)


def configure_tracing(service_name: str = "applied-ai") -> None:
    """
    Optionally configure OTel for local development with console export.

    In production, your service will already have OTel configured upstream.
    This helper is for running examples locally and seeing traces in the
    console. Call once at startup.
    """
    if not _OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry not installed. Spans will no-op. "
            "Install with: pip install opentelemetry-sdk"
        )
        return

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry configured with console exporter")
    except ImportError:
        logger.warning("opentelemetry-sdk not available; tracing disabled")
