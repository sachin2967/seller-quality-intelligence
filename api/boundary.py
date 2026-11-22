"""Bounded input and an operator-only boundary for the retrospective API."""

import hmac
import json
import os


class OperatorBoundary:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def reject(status, detail):
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps({"detail": detail}).encode(),
                }
            )

        expected = os.getenv("SELLER_API_KEY", "")
        if scope["path"] != "/live":
            if os.getenv("APP_ENV", "local") == "production" and len(expected) < 32:
                return await reject(
                    503,
                    "Production requires SELLER_API_KEY with at least 32 characters.",
                )
            if (
                scope["path"] not in {"/health", "/ready"}
                and expected
                and not hmac.compare_digest(
                    dict(scope.get("headers", [])).get(b"x-api-key", b""),
                    expected.encode(),
                )
            ):
                return await reject(401, "Valid X-API-Key required.")
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > 64 * 1024:
                return await reject(413, "Request body exceeds 64 KiB.")
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return await self.app(scope, bounded_receive, send)
