"""Optional real-model adapter for your own experiments. Never used by trials.

Every trial in the dungeon runs against the deterministic mocks in ``llm.py``.
When you want to point the code you wrote on Floors 9 to 11 at a real model,
this module gives you an adapter with the same ``complete()`` signature.

    export ANTHROPIC_API_KEY=...        # or `ant auth login`
    pip install anthropic

    from dungeon.artifacts.providers import AnthropicLLM
    llm = AnthropicLLM()                 # defaults to claude-opus-5
    llm.complete([Message.user("Say hello to the dungeon.")]).text

To add another provider, implement a class with the same ``complete`` method.
The dungeon deliberately ships only one adapter so that nothing here is a thin
copy of a vendor SDK; the point of these floors is the engineering around the
model, which is identical whichever model you use.
"""

from __future__ import annotations

from collections.abc import Sequence

from dungeon.artifacts.llm import (
    Completion,
    InvalidRequestError,
    LLMConnectionError,
    LLMTimeoutError,
    Message,
    RateLimitError,
    ServerError,
    ToolCall,
    ToolSpec,
    Usage,
)

DEFAULT_MODEL = "claude-opus-5"


class AnthropicLLM:
    """Adapter from the dungeon's ``Message``/``ToolSpec`` shapes to the Anthropic Messages API."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024, client=None):
        try:
            import anthropic  # noqa: F401 - imported lazily so the dungeon never requires it
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install anthropic to use AnthropicLLM") from exc
        import anthropic

        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    # ----------------------------------------------------------- conversion
    @staticmethod
    def _to_api_messages(messages: Sequence[Message]) -> tuple[str | None, list[dict]]:
        system_parts = [m.content for m in messages if m.role == "system" and m.content]
        system = "\n\n".join(system_parts) or None
        api: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "user":
                api.append({"role": "user", "content": m.content})
            elif m.role == "assistant":
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for call in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
                api.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
            elif m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                # Consecutive tool results belong in ONE user message.
                if api and api[-1]["role"] == "user" and isinstance(api[-1]["content"], list) \
                        and api[-1]["content"] and api[-1]["content"][0].get("type") == "tool_result":
                    api[-1]["content"].append(block)
                else:
                    api.append({"role": "user", "content": [block]})
            else:
                raise InvalidRequestError(f"unknown role {m.role!r}")
        return system, api

    @staticmethod
    def _to_api_tools(tools: Sequence[ToolSpec] | None) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in (tools or [])
        ]

    # ------------------------------------------------------------------ call
    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec] | None = None, *, max_tokens: int | None = None) -> Completion:
        anthropic = self._anthropic
        system, api_messages = self._to_api_messages(messages)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        api_tools = self._to_api_tools(tools)
        if api_tools:
            kwargs["tools"] = api_tools
        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            retry_after = None
            try:
                retry_after = float(exc.response.headers.get("retry-after", ""))
            except (TypeError, ValueError, AttributeError):
                pass
            raise RateLimitError(str(exc), retry_after=retry_after) from exc
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ServerError(str(exc)) from exc
            raise InvalidRequestError(str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMConnectionError(str(exc)) from exc

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
        stop = response.stop_reason or "end_turn"
        if stop not in ("end_turn", "tool_use", "max_tokens", "refusal"):
            stop = "end_turn"
        usage = Usage(getattr(response.usage, "input_tokens", 0), getattr(response.usage, "output_tokens", 0))
        return Completion(Message.assistant("".join(text_parts), calls), stop, usage)
