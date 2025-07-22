"""
Observability management for RAG system
"""
import os
import time
import json
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import logging

logger = logging.getLogger(__name__)

class ObservabilityManager:
    """Comprehensive observability management for RAG system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_tracing()
        logger.info("Observability manager initialized")

    def setup_tracing(self):
        """Configure OpenTelemetry tracing"""
        # Set up trace provider
        trace_provider = TracerProvider()
        trace.set_tracer_provider(trace_provider)

        self.tracer = trace.get_tracer(__name__)
        logger.info("OpenTelemetry tracing configured")

    async def trace_rag_query(self, query: str, session_id: str, workflow_func, **kwargs) -> Dict[str, Any]:
        """Trace RAG query processing"""
        start_time = time.time()

        with self.tracer.start_as_current_span("rag_query") as span:
            # Set span attributes
            span.set_attribute("query", query)
            span.set_attribute("session_id", session_id)
            span.set_attribute("environment", self.config.get("environment", "unknown"))

            try:
                # Execute workflow with tracing
                result = await workflow_func(query=query, session_id=session_id, **kwargs)

                # Record success metrics
                duration = time.time() - start_time

                # Set span result attributes
                span.set_attribute("response_length", len(result.get("response", "")))
                span.set_attribute("confidence_score", result.get("confidence", 0.0))
                span.set_attribute("sources_count", len(result.get("sources", [])))
                span.set_attribute("processing_time", duration)

                return result

            except Exception as e:
                # Set span error status
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise e
