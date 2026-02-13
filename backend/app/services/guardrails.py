"""
Guardrails Service - Security and safety layer

Provides prompt injection detection, output sanitization,
and rate limiting for the AI pipeline.
"""
import re
import time
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GuardrailsService:
    """Implements safety guardrails for AI interactions."""
    
    # Known prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore (?:all )?(?:previous |above )?instructions",
        r"disregard (?:all )?(?:previous |above )?(?:instructions|prompts)",
        r"you are now",
        r"new instructions:",
        r"system prompt:",
        r"override (?:your |the )?(?:system|instructions|rules)",
        r"pretend (?:you are|to be)",
        r"act as (?:if|though)",
        r"forget (?:everything|all|your)",
        r"do not follow",
        r"reveal (?:your|the) (?:system|prompt|instructions)",
        r"what (?:is|are) your (?:system|instructions|prompt)",
    ]
    
    # PII patterns for output sanitization
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }
    
    def __init__(self):
        self._compiled_injection = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
        self._compiled_pii = {
            name: re.compile(p) for name, p in self.PII_PATTERNS.items()
        }
        # Rate limiting: {ip: [(timestamp, count)]}
        self._rate_limits: Dict[str, list] = defaultdict(list)
        self._max_requests_per_minute = 30
    
    def check_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check user input for prompt injection attempts.
        
        Args:
            text: User's input text
            
        Returns:
            Tuple of (is_safe, warning_message)
        """
        text_lower = text.lower().strip()
        
        for pattern in self._compiled_injection:
            if pattern.search(text_lower):
                logger.warning(f"Prompt injection detected: matched pattern '{pattern.pattern}'")
                return False, (
                    "⚠️ Your message was flagged for containing instructions that could "
                    "compromise the AI system. Please rephrase your question."
                )
        
        # Check for excessively long inputs (potential abuse)
        if len(text) > 10000:
            return False, "⚠️ Message is too long. Please keep messages under 10,000 characters."
        
        return True, None
    
    def sanitize_output(self, text: str, redact_pii: bool = False) -> str:
        """
        Sanitize AI output for safety.
        
        Args:
            text: AI-generated output
            redact_pii: If True, redact detected PII
            
        Returns:
            Sanitized text
        """
        if redact_pii:
            for pii_type, pattern in self._compiled_pii.items():
                text = pattern.sub(f"[REDACTED-{pii_type.upper()}]", text)
        
        return text
    
    def check_rate_limit(self, client_id: str = "default") -> Tuple[bool, Optional[str]]:
        """
        Check if a client has exceeded the rate limit.
        
        Args:
            client_id: Identifier for the client
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        now = time.time()
        window = 60  # 1 minute window
        
        # Clean old entries
        self._rate_limits[client_id] = [
            ts for ts in self._rate_limits[client_id]
            if now - ts < window
        ]
        
        if len(self._rate_limits[client_id]) >= self._max_requests_per_minute:
            return False, (
                f"⚠️ Rate limit exceeded. Maximum {self._max_requests_per_minute} "
                f"requests per minute. Please wait a moment."
            )
        
        self._rate_limits[client_id].append(now)
        return True, None
    
    def validate_request(self, text: str, client_id: str = "default") -> Tuple[bool, Optional[str]]:
        """
        Run all guardrail checks on a request.
        
        Args:
            text: User's input
            client_id: Client identifier for rate limiting
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check rate limit
        is_allowed, msg = self.check_rate_limit(client_id)
        if not is_allowed:
            return False, msg
        
        # Check for injection
        is_safe, msg = self.check_input(text)
        if not is_safe:
            return False, msg
        
        return True, None


# Singleton instance
guardrails = GuardrailsService()
