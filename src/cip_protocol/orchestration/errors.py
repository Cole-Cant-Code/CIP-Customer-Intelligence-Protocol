"""Shared error-handling utility for tool implementations."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_and_return_tool_error(
    *, tool_name: str, exc: Exception, user_message: str
) -> str:
    """Log full exception details while returning a safe user-facing error."""
    logger.exception("Tool '%s' failed", tool_name, exc_info=exc)
    return user_message
