"""
Logging configuration for production RAG system
"""
import logging
import logging.config
import os
import sys
from datetime import datetime
from typing import Dict, Any

class ContextFilter(logging.Filter):
    """Add context information to log records"""

    def filter(self, record):
        record.service = "rag-system"
        record.environment = os.getenv("ENVIRONMENT", "development")
        return True

def setup_logging():
    """Setup logging configuration"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()

    if log_format == "json":
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                    "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
                },
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                }
            },
            "filters": {
                "context": {
                    "()": ContextFilter
                }
            },
            "handlers": {
                "default": {
                    "level": log_level,
                    "formatter": "json" if log_format == "json" else "standard",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "filters": ["context"]
                }
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": log_level,
                    "propagate": False
                }
            }
        }
    else:
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "default": {
                    "level": log_level,
                    "formatter": "standard",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout"
                }
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": log_level,
                    "propagate": False
                }
            }
        }

    logging.config.dictConfig(logging_config)

def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)
