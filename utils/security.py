"""
Security utilities for RAG system
"""
import re
import html
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class SecurityUtils:
    """Security utility functions"""

    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent XSS and injection attacks"""
        if not text:
            return ""

        # HTML escape
        sanitized = html.escape(text)

        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'<script.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*=',
        ]

        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)

        # Limit length
        return sanitized[:2000]

    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """Validate session ID format"""
        if not session_id:
            return False

        # Check format (alphanumeric + hyphens, reasonable length)
        pattern = r'^[a-zA-Z0-9\-_]{8,64}$'
        return bool(re.match(pattern, session_id))

    @staticmethod
    def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data for logging"""
        sensitive_keys = ['password', 'token', 'key', 'secret']
        masked_data = data.copy()

        for key, value in masked_data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 4:
                    masked_data[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                else:
                    masked_data[key] = '***'

        return masked_data
