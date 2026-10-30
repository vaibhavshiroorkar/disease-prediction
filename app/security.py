"""Administrator access and bounded request handling for a single worker."""

from __future__ import annotations

import math
import secrets
import time
from collections import OrderedDict, deque
from collections.abc import Callable

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.datastructures import MutableHeaders, State
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import config

admin_header = APIKeyHeader(name="X-API-Key", auto_error=False, scheme_name="AdminKey")


def require_admin(key: str | None = Depends(admin_header)) -> None:
    if not config.ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Administrator access is not configured")
    if not key or not secrets.compare_digest(key.encode(), config.ADMIN_API_KEY.encode()):
        raise HTTPException(
            status_code=401, detail="A valid administrator API key is required",
            headers={"WWW-Authenticate": "APIKey"},
        )


class SlidingWindow:
    def __init__(
        self, limit: int, window: int = 60, max_clients: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window = window
        self.max_clients = max_clients
        self.clock = clock
        self.clients: OrderedDict[str, deque[float]] = OrderedDict()

    def retry_after(self, client: str) -> int:
        now = self.clock()
        cutoff = now - self.window
        while self.clients and next(iter(self.clients.values()))[-1] <= cutoff:
            self.clients.popitem(last=False)
        if client not in self.clients:
            if len(self.clients) >= self.max_clients:
                return self.window
            self.clients[client] = deque()
        events = self.clients[client]
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= self.limit:
            return max(1, math.ceil(events[0] + self.window - now))
        events.append(now)
        self.clients.move_to_end(client)
        return 0


def client_address(scope: Scope, hops: int) -> str:
    """The address to rate limit on.

    Only the last `hops` X-Forwarded-For entries were written by proxies we
    trust. Anything earlier was supplied by the client, so it is ignored.
    """
    if hops > 0:
        forwarded: list[str] = []
        for name, value in scope.get("headers", []):
            if name == b"x-forwarded-for":
                forwarded += [part.strip() for part in value.decode("latin-1").split(",")]
        forwarded = [part for part in forwarded if part]
        if len(forwarded) >= hops:
            return forwarded[-hops]
    client = scope.get("client")
    return client[0] if client else "unknown"


class RequestGuard:
    def __init__(self, app: ASGIApp, state: State, max_body: int) -> None:
        self.app = app
        self.state = state
        self.max_body = max_body

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]

        async def send_private(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "no-referrer"
                headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
                headers["Cache-Control"] = "no-cache" if path.startswith("/static/") else "no-store"
                if path == "/":
                    headers["Content-Security-Policy"] = (
                        "default-src 'self'; script-src 'self'; style-src 'self'; "
                        "img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; "
                        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
                    )
            await send(message)

        limited = path.startswith(("/predictions", "/risk-predictions")) or (
            scope["method"] == "POST" and path.startswith("/predict/")
        )
        if limited:
            address = client_address(scope, config.TRUSTED_PROXY_HOPS)
            retry = self.state.prediction_limiter.retry_after(address)
            if retry:
                response = JSONResponse(
                    {"detail": "Too many requests. Try again shortly."}, status_code=429,
                    headers={"Retry-After": str(retry)},
                )
                await response(scope, receive, send_private)
                return

        if scope["method"] in {"POST", "PUT", "PATCH"}:
            chunks: list[bytes] = []
            size = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > self.max_body:
                    response = JSONResponse({"detail": "Request body is too large"}, status_code=413)
                    await response(scope, receive, send_private)
                    return
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            body = b"".join(chunks)
            delivered = False

            async def replay() -> Message:
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay, send_private)
            return
        await self.app(scope, receive, send_private)
