"""
Monitoring and metrics utilities
"""
import time
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import logging

logger = logging.getLogger(__name__)

class PrometheusMetrics:
    """Prometheus metrics collector"""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Request metrics
        self.requests_total = Counter(
            'rag_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'rag_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint'],
            registry=self.registry
        )

        # RAG-specific metrics
        self.llm_calls_total = Counter(
            'rag_llm_calls_total',
            'Total number of LLM calls',
            ['model', 'operation'],
            registry=self.registry
        )

        self.vector_searches_total = Counter(
            'rag_vector_searches_total',
            'Total number of vector searches',
            ['database'],
            registry=self.registry
        )

        self.errors_total = Counter(
            'rag_errors_total',
            'Total number of errors',
            ['endpoint', 'error_type'],
            registry=self.registry
        )

        self.active_sessions = Gauge(
            'rag_active_sessions',
            'Number of active sessions',
            registry=self.registry
        )

        logger.info("Prometheus metrics initialized")

    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics"""
        self.requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=status_code
        ).inc()
        self.request_duration.labels(endpoint=endpoint).observe(duration)

    def record_llm_call(self, model: str, operation: str):
        """Record LLM call metric"""
        self.llm_calls_total.labels(model=model, operation=operation).inc()

    def record_vector_search(self, database: str):
        """Record vector search metric"""
        self.vector_searches_total.labels(database=database).inc()

    def record_error(self, endpoint: str, error_type: str):
        """Record error metric"""
        self.errors_total.labels(endpoint=endpoint, error_type=error_type).inc()
