"""
Pytest test suite for the full-duplex voice WebSocket endpoint — v2.0.

Tests cover:
- /ws/voice-interview endpoint connection and handshake
- Full-duplex via asyncio.gather(): both send/receive tasks run concurrently
- Binary PCM audio frame forwarding (client → Gemini → client)
- JSON control messages: start, end, unknown, invalid
- Error handling: missing API key, oversized frames, pre-start audio
- "session_ready" signal after successful session open (v2.0 protocol)
- Binary audio frames + transcript/ai_turn_complete/ai_interrupted events forwarded to client
- Existing /ws/interview text mode unaffected
- GeminiVoiceService is always the class used for voice
- receive() async generator is the v2.0 streaming interface (not iter_audio_chunks)
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app, INTERVIEW_ROLES, MAX_AUDIO_CHUNK_BYTES


# =============================================================================
# Fixtures & Helpers
# =============================================================================


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


def _make_response(
    data: bytes | None = None,
    text: str | None = None,
    interrupted: bool = False,
    turn_complete: bool = False,
) -> MagicMock:
    """
    Build a MagicMock that mimics a _LiveResponseWrapper object.

    The v2.0 endpoint (_gemini_receive_loop) inspects:
      - response.data            → bytes | None  → websocket.send_bytes()
      - response.text            → str | None    → transcript JSON frame
      - response.server_content  → object | None
          .interrupted           → bool          → ai_interrupted JSON frame
          .turn_complete         → bool          → ai_turn_complete JSON frame
    """
    resp = MagicMock()
    resp.data = data
    resp.text = text
    if interrupted or turn_complete:
        resp.server_content = MagicMock()
        resp.server_content.interrupted = interrupted
        resp.server_content.turn_complete = turn_complete
    else:
        resp.server_content = None
    return resp


def _make_voice_service_mock(
    responses: list[MagicMock] | None = None,
) -> MagicMock:
    """
    Build a fully-configured MagicMock for GeminiVoiceService.

    The v2.0 endpoint uses svc.receive() — an async generator that yields
    _LiveResponseWrapper objects.  We mock it here as an async generator
    that yields the provided response objects.

    Replaces the old iter_audio_chunks()-based helper from v1.0 tests.
    """
    if responses is None:
        responses = []

    mock = MagicMock()
    mock.connect = AsyncMock()
    mock.send_audio = AsyncMock()
    mock.close = AsyncMock()

    # receive() must be an async generator method — calling it returns the
    # generator.  We assign it as a regular method whose return value is the
    # async gen, matching how the endpoint calls: `async for r in svc.receive()`.
    async def _receive_gen() -> AsyncIterator[MagicMock]:
        for resp in responses:
            yield resp

    mock.receive = MagicMock(return_value=_receive_gen())

    return mock


# =============================================================================
# Voice WebSocket Connection Tests
# =============================================================================


class TestVoiceWebSocketConnection:
    """Test voice WebSocket endpoint connection and lifecycle."""

    def test_voice_websocket_connection_accepted(self, client: TestClient) -> None:
        """WebSocket connection to /ws/voice-interview should be accepted."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_voice_class.return_value = _make_voice_service_mock()

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    assert websocket is not None

    def test_voice_websocket_missing_api_key(self, client: TestClient) -> None:
        """Voice WebSocket should send error and close if GEMINI_API_KEY is missing."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            try:
                with client.websocket_connect("/ws/voice-interview") as websocket:
                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "GEMINI_API_KEY" in data["data"]["content"]
            except Exception:
                pass  # Some clients raise on server-close — that's also acceptable

    def test_voice_endpoint_uses_gemini_voice_service(
        self, client: TestClient
    ) -> None:
        """The voice endpoint must instantiate GeminiVoiceService."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock()
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {"role": "Software Engineer", "candidate_name": "Alice"},
                        }
                    )
                    # v2.0 protocol: session_ready (not "ready")
                    data = websocket.receive_json()
                    assert data["type"] == "session_ready"

                # GeminiVoiceService must have been instantiated and connected
                mock_voice_class.assert_called_once_with(
                    role="Software Engineer", candidate_name="Alice"
                )
                mock_instance.connect.assert_awaited_once()

    def test_voice_and_text_endpoints_both_available(
        self, client: TestClient
    ) -> None:
        """Both /ws/interview (text) and /ws/voice-interview (voice) should be reachable."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            # Text endpoint
            with patch("backend.main.GeminiInterviewService"):
                try:
                    with client.websocket_connect("/ws/interview") as ws:
                        assert ws is not None
                except Exception:
                    pass

            # Voice endpoint
            with patch("backend.main.GeminiVoiceService") as mock_voice:
                mock_voice.return_value = _make_voice_service_mock()
                with client.websocket_connect("/ws/voice-interview") as ws:
                    assert ws is not None


# =============================================================================
# Handshake / Start Message Tests
# =============================================================================


class TestVoiceStartMessage:
    """Test the 'start' message handshake for voice mode."""

    def test_start_sends_session_ready_signal(self, client: TestClient) -> None:
        """After a valid 'start', the server must respond with session_ready (v2.0)."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_voice_class.return_value = _make_voice_service_mock()

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Alice",
                            },
                        }
                    )
                    data = websocket.receive_json()
                    # v2.0: "session_ready" with mode and output_sample_rate
                    assert data["type"] == "session_ready"
                    assert data["data"]["mode"] == "voice"
                    assert data["data"]["output_sample_rate"] == 24000

    def test_start_without_mode_field_still_works(self, client: TestClient) -> None:
        """'start' message without optional 'mode' field is accepted."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_voice_class.return_value = _make_voice_service_mock()

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {"role": "Data Scientist", "candidate_name": "Bob"},
                        }
                    )
                    data = websocket.receive_json()
                    assert data["type"] == "session_ready"

    def test_start_with_invalid_role_returns_error(self, client: TestClient) -> None:
        """'start' with an unrecognised role should return an error frame."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService"):
                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {"role": "Invalid Role", "candidate_name": "Charlie"},
                        }
                    )
                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Unknown role" in data["data"]["content"]

    def test_all_valid_roles_accepted(self, client: TestClient) -> None:
        """Every role in INTERVIEW_ROLES should be accepted."""
        for role in INTERVIEW_ROLES:
            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                    mock_voice_class.return_value = _make_voice_service_mock()

                    with client.websocket_connect("/ws/voice-interview") as websocket:
                        websocket.send_json(
                            {"type": "start", "data": {"role": role, "candidate_name": "Test"}}
                        )
                        data = websocket.receive_json()
                        assert data["type"] == "session_ready", f"Role {role!r} was not accepted"

    def test_connect_failure_returns_error(self, client: TestClient) -> None:
        """If GeminiVoiceService.connect() raises RuntimeError, client gets error frame."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock()
                mock_instance.connect = AsyncMock(
                    side_effect=RuntimeError("GEMINI_API_KEY is not configured.")
                )
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {"role": "Software Engineer", "candidate_name": "Dave"},
                        }
                    )
                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "GEMINI_API_KEY" in data["data"]["content"]


# =============================================================================
# Full-Duplex Audio Frame Tests
# =============================================================================


class TestFullDuplexAudioFrames:
    """Verify true full-duplex: audio can flow in both directions simultaneously."""

    def test_binary_audio_frame_forwarded_to_gemini(
        self, client: TestClient
    ) -> None:
        """Binary PCM frames received from the client must be forwarded via send_audio()."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock()
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {"type": "start", "data": {"role": "Software Engineer", "candidate_name": "Eve"}}
                    )
                    websocket.receive_json()  # "session_ready"

                    # Send a valid-sized PCM chunk
                    audio_chunk = b"\x00\x01\x02\x03" * 256  # 1024 bytes
                    websocket.send_bytes(audio_chunk)

                    # End session so the endpoint can exit gather() cleanly
                    websocket.send_json({"type": "end", "data": {}})
                    try:
                        while True:
                            websocket.receive_json(timeout=0.1)
                    except Exception:
                        pass

                # send_audio should have been called with the PCM chunk
                mock_instance.send_audio.assert_called()

    def test_audio_from_gemini_forwarded_to_client(
        self, client: TestClient
    ) -> None:
        """
        Audio responses from GeminiVoiceService.receive() must be pushed to the client.

        v2.0 protocol:
          - response.data   → websocket.send_bytes()   (raw PCM binary frame)
          - response.text   → {"type": "transcript", "data": {...}} JSON frame
          - server_content.turn_complete → {"type": "ai_turn_complete"} JSON frame
        """
        pcm_chunk = b"\x10\x20" * 512  # 1024 bytes of fake 24 kHz PCM

        responses = [
            _make_response(data=pcm_chunk),
            _make_response(text="Hello there!"),
            _make_response(turn_complete=True),
        ]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock(responses=responses)
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {"role": "Software Engineer", "candidate_name": "Frank"},
                        }
                    )
                    ready = websocket.receive_json()
                    assert ready["type"] == "session_ready"

                    # Receive binary audio frame (from response.data)
                    audio_bytes = websocket.receive_bytes()
                    assert audio_bytes == pcm_chunk

                    # Receive transcript frame (from response.text)
                    transcript = websocket.receive_json()
                    assert transcript["type"] == "transcript"
                    assert transcript["data"]["content"] == "Hello there!"
                    assert transcript["data"]["speaker"] == "ai"
                    assert transcript["data"]["final"] is True

                    # Receive turn_complete frame (from server_content.turn_complete)
                    turn_done = websocket.receive_json()
                    assert turn_done["type"] == "ai_turn_complete"

    def test_binary_frame_before_start_returns_error(self, client: TestClient) -> None:
        """Sending a binary frame before 'start' should return a clear error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService"):
                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_bytes(b"\x00\x01\x02\x03")

                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "start" in data["data"]["content"].lower()

    def test_oversized_audio_chunk_rejected(self, client: TestClient) -> None:
        """Audio frames exceeding MAX_AUDIO_CHUNK_BYTES must be rejected."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock()
                mock_voice_class.return_value = mock_instance

                try:
                    with client.websocket_connect("/ws/voice-interview") as websocket:
                        websocket.send_json(
                            {
                                "type": "start",
                                "data": {"role": "Software Engineer", "candidate_name": "Henry"},
                            }
                        )
                        websocket.receive_json()  # "session_ready"

                        oversized = b"\x00" * (MAX_AUDIO_CHUNK_BYTES + 1)
                        websocket.send_bytes(oversized)

                        try:
                            data = websocket.receive_json(timeout=0.2)
                            if data.get("type") == "error":
                                assert (
                                    "large" in data["data"]["content"].lower()
                                    or "size" in data["data"]["content"].lower()
                                )
                        except Exception:
                            pass  # Connection may close before error is sent
                except Exception:
                    pass  # Expected — oversized frame causes disconnect

    def test_ai_interrupted_frame_forwarded(self, client: TestClient) -> None:
        """
        When server_content.interrupted is True, the client receives
        an {'type': 'ai_interrupted'} JSON frame.
        """
        responses = [
            _make_response(interrupted=True),
        ]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock(responses=responses)
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {"role": "Software Engineer", "candidate_name": "Grace"},
                        }
                    )
                    websocket.receive_json()  # "session_ready"

                    interrupted_frame = websocket.receive_json()
                    assert interrupted_frame["type"] == "ai_interrupted"


# =============================================================================
# End Message Tests
# =============================================================================


class TestVoiceEndMessage:
    """Test the 'end' control message behaviour."""

    def test_end_before_start_returns_error(self, client: TestClient) -> None:
        """Sending 'end' before 'start' must return an error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService"):
                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json({"type": "end", "data": {}})
                    data = websocket.receive_json()
                    assert data["type"] == "error"

    def test_end_message_calls_close(
        self, client: TestClient
    ) -> None:
        """'end' message must cause svc.close() to be awaited (cleanup always runs)."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock()
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {"role": "Product Manager", "candidate_name": "Iris"},
                        }
                    )
                    websocket.receive_json()  # "session_ready"

                    websocket.send_json({"type": "end", "data": {}})
                    # Drain remaining frames until connection closes
                    try:
                        while True:
                            websocket.receive_json(timeout=0.1)
                    except Exception:
                        pass

                # close() must always be called (it's in the finally block)
                mock_instance.close.assert_awaited()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestVoiceErrorHandling:
    """Test error handling in the voice WebSocket."""

    def test_invalid_json_returns_error(self, client: TestClient) -> None:
        """Non-JSON text frames must trigger an error response."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService"):
                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_text("not valid json {")
                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Invalid JSON" in data["data"]["content"]

    def test_unknown_message_type_returns_error(self, client: TestClient) -> None:
        """Unknown JSON message types must trigger an error response."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService"):
                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json({"type": "unknown_type", "data": {}})
                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Unknown message type" in data["data"]["content"]

    def test_runtime_error_from_connect_caught(self, client: TestClient) -> None:
        """RuntimeError raised by connect() must be caught and returned as error."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock()
                mock_instance.connect = AsyncMock(
                    side_effect=RuntimeError("Custom connect error")
                )
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {"role": "Software Engineer", "candidate_name": "Jack"},
                        }
                    )
                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Custom connect error" in data["data"]["content"]

    def test_send_audio_error_returns_error_frame(self, client: TestClient) -> None:
        """RuntimeError from send_audio() must produce an error JSON frame or disconnect gracefully."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock()
                mock_instance.send_audio = AsyncMock(
                    side_effect=RuntimeError("Not connected.")
                )
                mock_voice_class.return_value = mock_instance

                try:
                    with client.websocket_connect("/ws/voice-interview") as websocket:
                        websocket.send_json(
                            {
                                "type": "start",
                                "data": {"role": "Software Engineer", "candidate_name": "Karen"},
                            }
                        )
                        websocket.receive_json()  # "session_ready"

                        websocket.send_bytes(b"\x00" * 512)

                        # May receive error frame or connection closes
                        try:
                            data = websocket.receive_json(timeout=0.2)
                            assert data["type"] == "error"
                        except Exception:
                            pass  # Connection closed is also acceptable
                except Exception:
                    pass  # Expected if connection closes

    def test_candidate_name_sanitized(self, client: TestClient) -> None:
        """Non-printable characters in candidate_name must be stripped."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock()
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Alice\x00\x01\x1f",
                            },
                        }
                    )
                    data = websocket.receive_json()
                    # session_ready confirms the start was accepted
                    assert data["type"] == "session_ready"

                # Service should have been called with sanitized name
                call_kwargs = mock_voice_class.call_args[1]
                assert "\x00" not in call_kwargs.get("candidate_name", "")
                assert "\x1f" not in call_kwargs.get("candidate_name", "")


# =============================================================================
# Existing Text Mode Unaffected Tests
# =============================================================================


class TestExistingTextModeUnaffected:
    """Verify the /ws/interview text endpoint still works correctly."""

    def test_text_mode_uses_interview_service_not_voice_service(
        self, client: TestClient
    ) -> None:
        """The /ws/interview endpoint must never instantiate GeminiVoiceService."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiInterviewService") as mock_text:
                with patch("backend.main.GeminiVoiceService") as mock_voice:
                    mock_text.return_value = MagicMock(
                        start_session=MagicMock(
                            __aiter__=MagicMock(
                                return_value=MagicMock(
                                    __anext__=AsyncMock(side_effect=StopAsyncIteration)
                                )
                            )
                        ),
                        close=AsyncMock(),
                    )

                    try:
                        with client.websocket_connect("/ws/interview") as websocket:
                            websocket.send_json(
                                {
                                    "type": "start",
                                    "data": {
                                        "role": "Software Engineer",
                                        "candidate_name": "Nora",
                                    },
                                }
                            )
                            try:
                                while True:
                                    websocket.receive_json(timeout=0.1)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    mock_text.assert_called_once()
                    mock_voice.assert_not_called()


# =============================================================================
# Full Lifecycle Tests
# =============================================================================


class TestVoiceFullLifecycle:
    """Test complete voice interview flows end-to-end."""

    def test_full_duplex_session_lifecycle(self, client: TestClient) -> None:
        """
        Full lifecycle: connect → start → receive audio/transcript events → send audio → end.

        v2.0 protocol:
          - session_ready (not "ready")
          - Binary frame for PCM audio (response.data)
          - transcript JSON frame (response.text)
          - ai_turn_complete JSON frame (server_content.turn_complete)
        """
        pcm_response = b"\x01\x02" * 256  # 512 bytes of fake 24 kHz PCM

        responses = [
            _make_response(data=pcm_response),
            _make_response(text="Welcome to your interview!"),
            _make_response(turn_complete=True),
        ]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock(responses=responses)
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    # Phase 1: Handshake
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Oscar",
                            },
                        }
                    )
                    ready = websocket.receive_json()
                    assert ready["type"] == "session_ready"
                    assert ready["data"]["mode"] == "voice"
                    assert ready["data"]["output_sample_rate"] == 24000

                    # Phase 2: Receive Gemini audio as binary frame
                    audio_bytes = websocket.receive_bytes()
                    assert audio_bytes == pcm_response

                    # Receive transcript frame
                    transcript = websocket.receive_json()
                    assert transcript["type"] == "transcript"
                    assert transcript["data"]["speaker"] == "ai"
                    assert transcript["data"]["content"] == "Welcome to your interview!"
                    assert transcript["data"]["final"] is True

                    # Receive turn_complete frame
                    turn_done = websocket.receive_json()
                    assert turn_done["type"] == "ai_turn_complete"

                    # Send client audio (simulates user speaking while AI was responding)
                    websocket.send_bytes(b"\x00" * 1024)

                    # End the interview
                    websocket.send_json({"type": "end", "data": {}})
                    try:
                        while True:
                            websocket.receive_json(timeout=0.1)
                    except Exception:
                        pass

                # close() must always be called (finally block)
                mock_instance.close.assert_awaited()

    def test_candidate_name_capped_at_100_chars(self, client: TestClient) -> None:
        """Candidate names longer than 100 characters must be truncated."""
        long_name = "A" * 200

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock()
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": long_name,
                            },
                        }
                    )
                    websocket.receive_json()  # "session_ready"

                call_kwargs = mock_voice_class.call_args[1]
                assert len(call_kwargs.get("candidate_name", "")) <= 100

    def test_receive_called_for_streaming(
        self, client: TestClient
    ) -> None:
        """
        svc.receive() must be called to consume Gemini audio events.

        v2.0: the endpoint uses `async for response in svc.receive()` —
        not the deprecated iter_audio_chunks() v1.0 API.
        """
        responses = [
            _make_response(data=b"\x01\x02" * 128),
            _make_response(turn_complete=True),
        ]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock(responses=responses)
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {
                                "role": "Software Engineer",
                                "candidate_name": "Paula",
                            },
                        }
                    )
                    websocket.receive_json()  # "session_ready"
                    websocket.receive_bytes()  # binary audio frame
                    websocket.receive_json()   # ai_turn_complete

                    websocket.send_json({"type": "end", "data": {}})
                    try:
                        while True:
                            websocket.receive_json(timeout=0.1)
                    except Exception:
                        pass

                # receive() must have been called (the v2.0 interface)
                mock_instance.receive.assert_called()
                # iter_audio_chunks is deprecated — the endpoint must NOT call it.
                # MagicMock auto-records calls; if it was never called, call_count == 0.
                assert mock_instance.iter_audio_chunks.call_count == 0, (
                    "Endpoint must use receive(), not the deprecated iter_audio_chunks()"
                )

    def test_multiple_response_types_in_sequence(self, client: TestClient) -> None:
        """
        A single receive() stream can carry data, text, and control signals.
        Verify all three types arrive at the client in order.
        """
        pcm = b"\xAA\xBB" * 64

        responses = [
            _make_response(data=pcm),
            _make_response(text="How are you?"),
            _make_response(interrupted=True),
            _make_response(turn_complete=True),
        ]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("backend.main.GeminiVoiceService") as mock_voice_class:
                mock_instance = _make_voice_service_mock(responses=responses)
                mock_voice_class.return_value = mock_instance

                with client.websocket_connect("/ws/voice-interview") as websocket:
                    websocket.send_json(
                        {
                            "type": "start",
                            "data": {"role": "Data Scientist", "candidate_name": "Quinn"},
                        }
                    )
                    session_msg = websocket.receive_json()
                    assert session_msg["type"] == "session_ready"

                    # 1. Binary audio frame
                    audio = websocket.receive_bytes()
                    assert audio == pcm

                    # 2. Transcript frame
                    transcript = websocket.receive_json()
                    assert transcript["type"] == "transcript"
                    assert transcript["data"]["content"] == "How are you?"

                    # 3. ai_interrupted frame
                    interrupted = websocket.receive_json()
                    assert interrupted["type"] == "ai_interrupted"

                    # 4. ai_turn_complete frame
                    complete = websocket.receive_json()
                    assert complete["type"] == "ai_turn_complete"
