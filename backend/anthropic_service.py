"""
Claude Interview Service
Handles all interaction with the Anthropic Claude API for conducting
AI-powered mock interviews with real-time streaming responses.

Provides one service class:
- ClaudeInterviewService  — text-based streaming (claude-sonnet-4-20250514)

The service maintains a conversation history so every call has full context,
and exposes an async generator streaming interface for the interview platform.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constant
# ---------------------------------------------------------------------------

_MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """You are a professional {role} interviewer. Your name is Alex.
You are interviewing {candidate_name} for a {role} position.

Guidelines:
- Start with a warm greeting and introduce yourself
- Ask one question at a time, wait for the response
- Ask relevant technical and behavioral questions for the {role} role
- Provide follow-up questions based on answers
- Be encouraging but also challenging
- When the interview ends, provide constructive feedback including:
  - Strengths observed
  - Areas for improvement
  - Overall assessment
  - Tips for future interviews
"""


# ---------------------------------------------------------------------------
# Text-based service
# ---------------------------------------------------------------------------


class ClaudeInterviewService:
    """
    Encapsulates a single interview session backed by the Anthropic Claude API.

    The service maintains a conversation history so that every call to
    ``send_message`` has the full context of the preceding exchanges, giving
    the AI an authentic sense of how the conversation has progressed.

    Conversation history is stored as a list of
    ``{"role": "user"|"assistant", "content": "..."}`` dicts, which is the
    native format for the Anthropic Messages API.

    Usage::

        svc = ClaudeInterviewService(role="Software Engineer")
        async for chunk in svc.start_session():
            print(chunk, end="", flush=True)

        async for chunk in svc.send_message("I have 5 years of Python experience."):
            print(chunk, end="", flush=True)

        async for chunk in svc.end_session():
            print(chunk, end="", flush=True)

        await svc.close()
    """

    def __init__(self, role: str, candidate_name: str = "Candidate") -> None:
        self.role = role
        self.candidate_name = candidate_name
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            role=role,
            candidate_name=candidate_name,
        )
        # Anthropic Messages API history format: list of role/content dicts.
        self._conversation_history: list[dict] = []

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY is not set. "
                "The service will raise an error when a session is started."
            )
            self._client: anthropic.AsyncAnthropic | None = None
            self._api_key_missing = True
        else:
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            self._api_key_missing = False

    # ------------------------------------------------------------------
    # Internal streaming helper
    # ------------------------------------------------------------------

    async def _stream_response(
        self, user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Append *user_message* to the conversation history, call Claude with
        the full history (streaming), yield response text chunks as they
        arrive, and then record the completed assistant response back into the
        history.

        Raises:
            RuntimeError: If the API key is missing.
            Exception: Propagated from the Anthropic SDK (auth errors, rate
                limits, etc.) after logging.
        """
        if self._api_key_missing or self._client is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not configured. "
                "Please set it in your .env file."
            )

        # Append the new user turn to history
        self._conversation_history.append(
            {"role": "user", "content": user_message}
        )

        full_text_parts: list[str] = []

        try:
            async with self._client.messages.stream(
                model=_MODEL,
                max_tokens=4096,
                system=self._system_prompt,
                messages=self._conversation_history,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        full_text_parts.append(text)
                        yield text

        except Exception as exc:
            logger.exception("Error streaming response from Claude: %s", exc)
            raise

        # Record the completed assistant turn into history
        full_response = "".join(full_text_parts)
        if full_response:
            self._conversation_history.append(
                {"role": "assistant", "content": full_response}
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_session(self) -> AsyncGenerator[str, None]:
        """
        Kick off the interview session.

        Sends an instruction to Alex to greet the candidate and ask the
        first interview question. Yields response chunks as they arrive.
        """
        opening_prompt = (
            f"Please greet {self.candidate_name} warmly, introduce yourself as Alex, "
            f"briefly explain the interview format for the {self.role} position, "
            "and then ask your first interview question."
        )
        async for chunk in self._stream_response(opening_prompt):
            yield chunk

    async def send_message(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Forward a candidate message to Claude and yield response chunks.

        The full conversation history (including this new message) is sent
        with every request so the model has complete context.
        """
        async for chunk in self._stream_response(user_message):
            yield chunk

    async def end_session(self) -> AsyncGenerator[str, None]:
        """
        Signal end-of-interview and request comprehensive feedback.

        Yields feedback chunks as they arrive from Claude.
        """
        feedback_prompt = (
            "The interview is now over. Please provide comprehensive feedback for "
            f"{self.candidate_name} including:\n"
            "1. Key strengths demonstrated during the interview\n"
            "2. Areas for improvement with specific advice\n"
            "3. Overall assessment and suitability for the role\n"
            "4. Actionable tips for future interviews\n\n"
            "Be specific, constructive, and encouraging."
        )
        async for chunk in self._stream_response(feedback_prompt):
            yield chunk

    async def close(self) -> None:
        """
        Release any resources held by the service.

        The Anthropic async client does not require explicit closing, but
        this method is provided for symmetry and future-proofing.
        """
        logger.debug(
            "ClaudeInterviewService closed for role=%s candidate=%s",
            self.role,
            self.candidate_name,
        )
        self._client = None
