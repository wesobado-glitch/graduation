"""
Temporary debug middleware to log incoming request details on Railway.
Remove this after debugging is complete.
"""
import logging

logger = logging.getLogger(__name__)


class RequestDebugMiddleware:
    """Logs request method, content type, body size, and key headers for debugging."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        logger.warning(
            "[DEBUG-MIDDLEWARE] %s %s | Content-Type: %s | Body length: %d | "
            "X-Forwarded-Proto: %s | X-Forwarded-For: %s",
            request.method,
            request.get_full_path(),
            request.content_type,
            len(request.body),
            request.META.get('HTTP_X_FORWARDED_PROTO', 'N/A'),
            request.META.get('HTTP_X_FORWARDED_FOR', 'N/A'),
        )
        response = self.get_response(request)
        return response
