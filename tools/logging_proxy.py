"""Transparent logging proxy for debugging what a client actually sends to vLLM.

Listens on :8001 and forwards to :8000, dumping each request/response to stdout.
"""

import json
import sys
import time

from aiohttp import ClientSession, ClientTimeout, web

UPSTREAM = "http://127.0.0.1:8000"
REDACT = {"authorization", "api-key", "x-api-key", "cookie"}


def _dump(label: str, payload: object) -> None:
    print(f"\n{'=' * 70}\n{label}  {time.strftime('%H:%M:%S')}\n{'=' * 70}", flush=True)
    print(payload, flush=True)


async def handler(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    headers = {
        k: ("<redacted>" if k.lower() in REDACT else v) for k, v in request.headers.items()
    }
    _dump(f"--> {request.method} {request.path_qs}", json.dumps(headers, indent=2))

    if body:
        try:
            parsed = json.loads(body)
            summary = {k: v for k, v in parsed.items() if k != "messages"}
            summary["messages"] = [
                {
                    "role": m.get("role"),
                    "len": len(str(m.get("content"))),
                    "preview": str(m.get("content"))[:160],
                }
                for m in parsed.get("messages", [])
            ]
            _dump("--> BODY", json.dumps(summary, indent=2)[:6000])
        except json.JSONDecodeError:
            _dump("--> BODY (raw)", body[:2000])

    fwd = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    timeout = ClientTimeout(total=None, sock_read=None)
    async with ClientSession(timeout=timeout) as session:
        async with session.request(
            request.method, UPSTREAM + request.path_qs, headers=fwd, data=body
        ) as upstream:
            _dump(f"<-- {upstream.status}", json.dumps(dict(upstream.headers), indent=2))
            out = web.StreamResponse(status=upstream.status, headers={
                k: v for k, v in upstream.headers.items()
                if k.lower() not in {"content-length", "content-encoding", "transfer-encoding"}
            })
            await out.prepare(request)
            total = 0
            first = None
            async for chunk in upstream.content.iter_any():
                if first is None:
                    first = time.time()
                    _dump("<-- FIRST CHUNK", chunk[:1500])
                total += len(chunk)
                await out.write(chunk)
            await out.write_eof()
            _dump("<-- DONE", f"{total} bytes streamed")
            return out


app = web.Application(client_max_size=1024**3)
app.router.add_route("*", "/{tail:.*}", handler)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    web.run_app(app, host="0.0.0.0", port=port)
