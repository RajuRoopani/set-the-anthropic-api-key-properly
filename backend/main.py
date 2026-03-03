"""
AI Interview Platform — FastAPI Application
===========================================
Routes
------
  GET  /                       Serve the frontend SPA (static/index.html)
  GET  /api/health             Health check endpoint
  GET  /api/config/roles       Return the list of available interview roles
  WS   /ws/interview           Real-time text interview session via WebSocket
  WS   /ws/voice-interview     Voice mode placeholder (not available)

WebSocket message protocol — text mode (/ws/interview)
-------------------------------------------------------
Client → Server  (JSON)
  {"type": "start",   "data": {"role": "<role>", "candidate_name": "<name>"}}
  {"type": "message", "data": {"content": "<user utterance>"}}
  {"type": "end",     "data": {}}

Server → Client  (JSON)
  {"type": "chunk",    "data": {"content": "<partial text>", "done": false}}
  {"type": "response", "data": {"content": "<full text>",    "done": true}}
  {"type": "error",    "data": {"content": "<error message>"}}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.anthropic_service import ClaudeInterviewService

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()  # Load .env if present — harmless when vars already in environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

if not os.getenv("ANTHROPIC_API_KEY", "").strip():
    logger.warning(
        "⚠️  ANTHROPIC_API_KEY is not set. "
        "The app will start, but interview sessions will not work until "
        "the key is supplied via the ANTHROPIC_API_KEY environment variable."
    )

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # /workspace/interview_platform/
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="AI Interview Platform",
    description="Practice job interviews with a real-time AI interviewer powered by Claude.",
    version="2.0.0",
)

# Mount static assets (CSS, JS, images).  The directory must exist for the
# mount to succeed — create it lazily so the app starts even before the
# UX engineer delivers assets.
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Available interview roles
# ---------------------------------------------------------------------------

INTERVIEW_ROLES: list[str] = [
    "Software Engineer",
    "Data Scientist",
    "Product Manager",
    "System Design",
]

# ---------------------------------------------------------------------------
# Constants — exported so tests can reference them directly
# ---------------------------------------------------------------------------

# Maximum allowed length for a single candidate message.
# Aligned with the frontend HTML maxlength attribute.
MAX_MESSAGE_LENGTH: int = 2000

# Minimal fallback HTML returned when index.html has not been deployed yet.
_FALLBACK_HTML: str = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Interview Platform</title>
</head>
<body>
  <h1>AI Interview Platform</h1>
  <p>The frontend assets are not yet available.  Backend is running.</p>
  <p>Visit <a href="/docs">/docs</a> for the API documentation.</p>
</body>
</html>"""

# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False, response_model=None)
async def serve_index():  # type: ignore[return]
    """Serve the single-page application entry point.

    Falls back to a minimal inline HTML page when index.html has not been
    deployed, so the backend can be tested independently without causing a
    500 error.
    """
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        # Return a safe inline fallback — never attempt FileResponse on a
        # path that does not exist, as that raises a FileNotFoundError.
        return HTMLResponse(content=_FALLBACK_HTML, status_code=200)
    return FileResponse(path=str(index_path), media_type="text/html")


@app.get("/api/health", include_in_schema=True)
async def health_check() -> JSONResponse:
    """Simple health check endpoint.

    Returns ``{"status": "ok"}`` with HTTP 200.  Used by the frontend,
    load-balancers, and tests to verify the server is reachable.
    """
    return JSONResponse(content={"status": "ok"})


@app.get("/api/config/roles", response_model=list[str])
async def get_roles() -> list[str]:
    """Return the list of available interview roles."""
    return INTERVIEW_ROLES


# ---------------------------------------------------------------------------
# WebSocket helper utilities
# ---------------------------------------------------------------------------


async def _send_json(ws: WebSocket, payload: dict[str, Any]) -> None:
    """Serialize *payload* to JSON and send it over *ws*."""
    await ws.send_text(json.dumps(payload))


async def _send_error(ws: WebSocket, message: str) -> None:
    """Send a structured error message to the client."""
    await _send_json(ws, {"type": "error", "data": {"content": message}})


async def _stream_to_ws(
    ws: WebSocket,
    generator,  # AsyncGenerator[str, None]
) -> str:
    """
    Drive an async generator that yields text chunks, forwarding each chunk
    to the WebSocket client as it arrives.

    Returns the concatenated full response text so the caller can send the
    final `response` frame.
    """
    accumulated: list[str] = []

    async for chunk in generator:
        if chunk:
            accumulated.append(chunk)
            await _send_json(
                ws,
                {
                    "type": "chunk",
                    "data": {"content": chunk, "done": False},
                },
            )

    full_response = "".join(accumulated)
    await _send_json(
        ws,
        {
            "type": "response",
            "data": {"content": full_response, "done": True},
        },
    )
    return full_response


# ---------------------------------------------------------------------------
# Text WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/interview")
async def interview_ws(websocket: WebSocket) -> None:
    """
    Real-time text interview session endpoint.

    Lifecycle
    ---------
    1. Accept connection (reject if API key is missing).
    2. Wait for a `start` message — instantiate ``ClaudeInterviewService``
       and stream the opening greeting.
    3. Process `message` messages — stream AI responses.
    4. On `end` message — stream feedback, then close cleanly.
    5. Handle disconnections and errors gracefully throughout.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted from %s", websocket.client)

    # Guard: reject immediately if the API key is absent
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        await _send_error(
            websocket,
            "ANTHROPIC_API_KEY is not configured on the server. "
            "Please contact the administrator.",
        )
        await websocket.close(code=1011)
        return

    interview_service: ClaudeInterviewService | None = None

    try:
        while True:
            raw = await websocket.receive_text()

            # ----------------------------------------------------------------
            # Parse incoming message
            # ----------------------------------------------------------------
            try:
                msg = json.loads(raw)
                msg_type: str = msg.get("type", "")
                msg_data: dict[str, Any] = msg.get("data", {})
            except (json.JSONDecodeError, AttributeError):
                await _send_error(websocket, "Invalid JSON message format.")
                continue

            # ----------------------------------------------------------------
            # Handle: start
            # ----------------------------------------------------------------
            if msg_type == "start":
                role: str = str(msg_data.get("role", "Software Engineer"))
                # Sanitize candidate_name: strip non-printable/control
                # characters and cap at 100 chars to prevent prompt-injection
                # payloads from escaping the Claude system prompt context.
                raw_name: str = str(msg_data.get("candidate_name", "Candidate"))
                candidate_name: str = (
                    "".join(ch for ch in raw_name if ch.isprintable()).strip()[:100]
                    or "Candidate"
                )

                if role not in INTERVIEW_ROLES:
                    await _send_error(
                        websocket,
                        f"Unknown role '{role}'. "
                        f"Valid roles are: {', '.join(INTERVIEW_ROLES)}.",
                    )
                    continue

                # Clean up any existing session
                if interview_service is not None:
                    await interview_service.close()

                interview_service = ClaudeInterviewService(
                    role=role,
                    candidate_name=candidate_name,
                )
                logger.info(
                    "Interview started: role=%s candidate=%s",
                    role,
                    candidate_name,
                )

                try:
                    await _stream_to_ws(websocket, interview_service.start_session())
                except RuntimeError as exc:
                    await _send_error(websocket, str(exc))
                except Exception as exc:
                    logger.exception("Error during session start: %s", exc)
                    await _send_error(
                        websocket,
                        _classify_anthropic_error(exc),
                    )

            # ----------------------------------------------------------------
            # Handle: message
            # ----------------------------------------------------------------
            elif msg_type == "message":
                if interview_service is None:
                    await _send_error(
                        websocket,
                        "No active interview session. Please send a 'start' message first.",
                    )
                    continue

                user_content: str = msg_data.get("content", "").strip()
                if not user_content:
                    await _send_error(websocket, "Message content cannot be empty.")
                    continue

                # Enforce server-side length cap (HTML maxlength is client-only
                # and trivially bypassed via a raw WebSocket client).
                if len(user_content) > MAX_MESSAGE_LENGTH:
                    await _send_error(
                        websocket,
                        f"Message too long. Maximum length is {MAX_MESSAGE_LENGTH} characters.",
                    )
                    continue

                try:
                    await _stream_to_ws(
                        websocket, interview_service.send_message(user_content)
                    )
                except RuntimeError as exc:
                    await _send_error(websocket, str(exc))
                except Exception as exc:
                    logger.exception("Error during message handling: %s", exc)
                    await _send_error(websocket, _classify_anthropic_error(exc))

            # ----------------------------------------------------------------
            # Handle: end
            # ----------------------------------------------------------------
            elif msg_type == "end":
                if interview_service is None:
                    await _send_error(
                        websocket,
                        "No active interview session to end.",
                    )
                    continue

                try:
                    await _stream_to_ws(
                        websocket, interview_service.end_session()
                    )
                except RuntimeError as exc:
                    await _send_error(websocket, str(exc))
                except Exception as exc:
                    logger.exception("Error during session end: %s", exc)
                    await _send_error(websocket, _classify_anthropic_error(exc))
                finally:
                    await interview_service.close()
                    interview_service = None

                # Signal graceful close to the client
                await websocket.close(code=1000)
                return

            # ----------------------------------------------------------------
            # Unknown message type
            # ----------------------------------------------------------------
            else:
                await _send_error(
                    websocket,
                    f"Unknown message type '{msg_type}'. "
                    "Expected one of: start, message, end.",
                )

    except WebSocketDisconnect:
        logger.info(
            "WebSocket client disconnected from %s", websocket.client
        )
    except Exception as exc:
        logger.exception("Unexpected WebSocket error: %s", exc)
        try:
            await _send_error(websocket, "An unexpected server error occurred.")
        except Exception:
            pass  # Connection may already be dead
    finally:
        if interview_service is not None:
            await interview_service.close()
            logger.debug("Interview service cleaned up on disconnect.")


# ---------------------------------------------------------------------------
# Voice WebSocket endpoint — not available with Anthropic provider
# ---------------------------------------------------------------------------


@app.websocket("/ws/voice-interview")
async def voice_interview_ws(websocket: WebSocket) -> None:
    """
    Voice interview endpoint — not available with the Anthropic Claude provider.

    Accepts the connection, sends an informative error message, and closes
    with code 1000 (normal closure).  Clients should fall back to text mode.
    """
    await websocket.accept()
    await _send_json(
        websocket,
        {
            "type": "error",
            "data": {
                "content": (
                    "Voice mode is not available with the current AI provider. "
                    "Please use text mode instead."
                )
            },
        },
    )
    await websocket.close(code=1000)


# ---------------------------------------------------------------------------
# Anthropic error classification
# ---------------------------------------------------------------------------


def _classify_anthropic_error(exc: Exception) -> str:
    """
    Map common Anthropic / networking exceptions to user-friendly messages.

    Handles standard HTTP status codes and common Anthropic API error keywords
    including authentication, permission denied, quota, rate-limiting,
    overloaded, timeout, and model-not-found patterns.
    """
    error_str = str(exc).lower()

    if (
        "api_key" in error_str
        or "api key" in error_str
        or "401" in error_str
        or "permission" in error_str
        or "denied" in error_str
    ):
        return (
            "Authentication failed. Please check that a valid ANTHROPIC_API_KEY "
            "is configured on the server."
        )
    if "quota" in error_str or "rate" in error_str or "429" in error_str:
        return (
            "The AI service is temporarily rate-limited. "
            "Please wait a few seconds and try again."
        )
    if "overloaded" in error_str or "529" in error_str:
        return (
            "The AI service is currently overloaded. "
            "Please wait a moment and try again."
        )
    if "timeout" in error_str or "deadline" in error_str:
        return "The request to the AI service timed out. Please try again."
    if "not found" in error_str or "404" in error_str:
        return "The requested AI model was not found. Please contact the administrator."

    return "An error occurred while communicating with the AI service."


# ---------------------------------------------------------------------------
# Entry point (development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
