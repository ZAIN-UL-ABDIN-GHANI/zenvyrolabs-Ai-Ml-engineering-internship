"""
services/aiService.py

Local-LLM-backed debate service for the Autonomous AI Debate Chamber.

Defines DebateConductor, which:
  * Holds two fixed personas -- Agent A (logical/evidence-driven) and
    Agent B (aggressive/persuasive) -- as required by the assignment spec.
  * Talks to a local LLM over HTTP. Ollama is the default/preferred
    provider; LM Studio and GPT4All are supported as configurable
    OpenAI-compatible alternatives (Task 2's "if Ollama is unavailable"
    requirement) via environment variables, so no code changes are needed
    to switch providers.
  * Maintains per-topic conversation memory (Task 3) so later rounds can
    reference every prior argument and rebuttal, and exposes a full
    multi-round debate engine (Task 4).

Configuration (all optional, all read from environment variables so no
secrets or endpoints are hardcoded):
  LLM_PROVIDER          "ollama" (default) | "lmstudio" | "gpt4all"
  LLM_MODEL             model name, e.g. "mistral", "llama3", "phi3", "gemma3"
                         (default: "mistral")
  LLM_BASE_URL          override the provider's default endpoint
  LLM_REQUEST_TIMEOUT   seconds before a generation request times out (default: 60)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class LLMConnectionError(Exception):
    """Raised when the configured local LLM provider cannot be reached at all
    (offline, wrong port, timeout establishing a connection)."""


class LLMGenerationError(Exception):
    """Raised when the provider responds, but with an error status, malformed
    JSON, an unexpected response shape, or an empty completion."""


_DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434/api/generate",
    "lmstudio": "http://localhost:1234/v1/chat/completions",
    "gpt4all": "http://localhost:4891/v1/chat/completions",
}


class DebateConductor:
    """
    Orchestrates a multi-round debate between two AI personas backed by a
    local LLM, with per-topic memory so later turns can reference the full
    prior exchange.
    """

    AGENT_A_SYSTEM_PROMPT = (
        "You are Agent A. You debate using logic, evidence, statistics and structured "
        "reasoning. Stay calm and precise, back claims with concrete reasoning, and "
        "directly refute weak logic in your opponent's argument. Keep responses to "
        "2-4 focused paragraphs."
    )
    AGENT_B_SYSTEM_PROMPT = (
        "You are Agent B. You aggressively attack weak arguments while remaining "
        "respectful. Be persuasive and emotionally compelling, challenge the "
        "assumptions behind your opponent's reasoning, and look for logical flaws. "
        "Keep responses to 2-4 focused paragraphs."
    )

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> None:
        """
        Args:
            provider: "ollama" | "lmstudio" | "gpt4all". Falls back to the
                LLM_PROVIDER env var, then "ollama".
            model: model name understood by the provider. Falls back to
                LLM_MODEL env var, then "mistral".
            base_url: full endpoint URL. Falls back to LLM_BASE_URL env var,
                then the provider's documented default local port.
            request_timeout: seconds before giving up on a single generation
                call. Falls back to LLM_REQUEST_TIMEOUT env var, then 60.
        """
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "ollama")).lower()
        self.model = model or os.environ.get("LLM_MODEL", "mistral")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or _DEFAULT_BASE_URLS.get(
            self.provider, _DEFAULT_BASE_URLS["ollama"]
        )
        self.request_timeout = float(
            request_timeout or os.environ.get("LLM_REQUEST_TIMEOUT", 60)
        )

        # Per-topic memory: topic -> ordered list of {"agent": "A"|"B", "response": str}.
        # Keyed by topic (rather than a single flat list) so multiple debate
        # topics don't bleed into each other's context.
        self.debate_history: Dict[str, List[Dict[str, str]]] = {}

        logger.info(
            "DebateConductor ready | provider=%s model=%s url=%s timeout=%ss",
            self.provider, self.model, self.base_url, self.request_timeout,
        )

    # ------------------------------------------------------------------
    # Memory (Task 3)
    # ------------------------------------------------------------------

    def reset_topic(self, topic: str) -> None:
        """Clear stored memory for a topic, starting a fresh debate session."""
        self.debate_history[topic] = []
        logger.info("Memory reset for topic=%r", topic)

    def get_history(self, topic: str) -> List[Dict[str, str]]:
        """Return a copy of the full turn history for a topic (empty list if unseen)."""
        return list(self.debate_history.get(topic, []))

    def get_round_number(self, topic: str) -> int:
        """Number of turns recorded so far for a topic."""
        return len(self.debate_history.get(topic, []))

    def _record_turn(self, topic: str, agent: str, response: str) -> None:
        self.debate_history.setdefault(topic, []).append({"agent": agent, "response": response})

    def _build_prompt(self, topic: str, speaker: str, opponent_last_message: str = "") -> str:
        """
        Build the debate prompt following the assignment's required template:

            Topic:
            Previous Debate:
            Agent A:
            ...
            Agent B:
            ...
            Now produce the next rebuttal.

        Falls back to an opening-statement prompt when there is no prior
        history at all.
        """
        history = self.debate_history.get(topic, [])
        lines = [f"Topic: {topic}", ""]

        if history:
            lines.append("Previous Debate:")
            for turn in history:
                label = "Agent A" if turn["agent"] == "A" else "Agent B"
                lines.append(f"{label}:")
                lines.append(turn["response"])
                lines.append("")

        # If the caller passed the opponent's latest message and it isn't
        # already the last recorded turn (e.g. the frontend sent it directly
        # rather than us having generated it ourselves), fold it in too.
        already_recorded = bool(history) and history[-1]["response"] == opponent_last_message
        if opponent_last_message and not already_recorded:
            opponent_label = "Agent B" if speaker == "A" else "Agent A"
            lines.append(f"{opponent_label}:")
            lines.append(opponent_last_message)
            lines.append("")

        if history or opponent_last_message:
            lines.append("Now produce the next rebuttal.")
        else:
            lines.append(
                "Provide your opening argument on this topic. Be clear, structured, "
                "and stay true to your persona."
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Agent turn generation (Task 2)
    # ------------------------------------------------------------------

    def generate_agent_a_response(self, topic: str, opponent_last_message: str = "") -> Dict[str, str]:
        """Generate Agent A's next turn (logical / evidence-driven persona)."""
        prompt = self._build_prompt(topic, "A", opponent_last_message)
        logger.info("Generating Agent A turn for topic=%r (round %d)", topic, self.get_round_number(topic) + 1)
        text = self._call_llm(self.AGENT_A_SYSTEM_PROMPT, prompt)
        self._record_turn(topic, "A", text)
        return {"agent": "A", "response": text}

    def generate_agent_b_response(self, topic: str, opponent_last_message: str = "") -> Dict[str, str]:
        """Generate Agent B's next turn (aggressive / persuasive persona)."""
        prompt = self._build_prompt(topic, "B", opponent_last_message)
        logger.info("Generating Agent B turn for topic=%r (round %d)", topic, self.get_round_number(topic) + 1)
        text = self._call_llm(self.AGENT_B_SYSTEM_PROMPT, prompt)
        self._record_turn(topic, "B", text)
        return {"agent": "B", "response": text}

    # ------------------------------------------------------------------
    # Debate engine (Task 4)
    # ------------------------------------------------------------------

    def run_full_debate(self, topic: str, rounds: int = 5) -> List[Dict[str, str]]:
        """
        Run an entire debate in one call:
          Agent A opening -> Agent B rebuttal -> alternating counters for the
          requested number of rounds -> a closing statement from each agent.

        Args:
            topic: the debate topic.
            rounds: total number of alternating rounds before final
                statements (default 5, per the assignment spec).

        Returns:
            The full transcript as a list of {"agent": "A"|"B", "response": str}
            turns, in the order they were generated. This is also what ends
            up in self.debate_history[topic] afterward.
        """
        self.reset_topic(topic)
        transcript: List[Dict[str, str]] = []

        opening = self.generate_agent_a_response(topic)
        transcript.append(opening)
        last_message = opening["response"]
        speaker = "B"

        for _ in range(max(rounds - 1, 0)):
            if speaker == "B":
                turn = self.generate_agent_b_response(topic, opponent_last_message=last_message)
            else:
                turn = self.generate_agent_a_response(topic, opponent_last_message=last_message)
            transcript.append(turn)
            last_message = turn["response"]
            speaker = "A" if speaker == "B" else "B"

        # Final statements: give each agent one closing turn.
        final_a = self.generate_agent_a_response(topic, opponent_last_message=last_message)
        transcript.append(final_a)
        final_b = self.generate_agent_b_response(topic, opponent_last_message=final_a["response"])
        transcript.append(final_b)

        return transcript

    # ------------------------------------------------------------------
    # LLM transport (Ollama / LM Studio / GPT4All)
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Dispatch to the configured provider and normalize errors into
        LLMConnectionError (provider unreachable) or LLMGenerationError
        (provider reachable but returned something unusable).
        """
        start = time.time()
        try:
            if self.provider == "ollama":
                text = self._call_ollama(system_prompt, user_prompt)
            else:
                # LM Studio and GPT4All both expose an OpenAI-compatible
                # /v1/chat/completions endpoint.
                text = self._call_openai_compatible(system_prompt, user_prompt)
        except requests.exceptions.ConnectTimeout as exc:
            raise LLMConnectionError(
                f"Timed out connecting to {self.provider} at {self.base_url}. Is it running?"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not reach {self.provider} at {self.base_url}. "
                f"For Ollama: run `ollama serve` and `ollama pull {self.model}`, then retry."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(
                f"Request to {self.provider} timed out after {self.request_timeout}s."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMGenerationError(f"Request to {self.provider} failed: {exc}") from exc

        elapsed = time.time() - start
        logger.info("LLM call via %s (%s) took %.2fs, %d chars returned", self.provider, self.model, elapsed, len(text))

        if not text or not text.strip():
            raise LLMGenerationError("LLM returned an empty response.")
        return text.strip()

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """POST to Ollama's /api/generate with stream disabled."""
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
        }
        resp = requests.post(self.base_url, json=payload, timeout=self.request_timeout)
        if resp.status_code != 200:
            # Covers "invalid model" (Ollama returns a non-200 with a message
            # like "model 'x' not found, try pulling it first") and other
            # server-side errors.
            raise LLMGenerationError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMGenerationError(f"Ollama returned non-JSON output: {exc}") from exc
        if "response" not in data:
            raise LLMGenerationError(f"Ollama response missing 'response' field: {data}")
        return data["response"]

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        """POST to an OpenAI-compatible /v1/chat/completions endpoint (LM Studio, GPT4All)."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        resp = requests.post(self.base_url, json=payload, timeout=self.request_timeout)
        if resp.status_code != 200:
            raise LLMGenerationError(f"{self.provider} returned HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMGenerationError(f"Unexpected {self.provider} response shape: {exc}") from exc
