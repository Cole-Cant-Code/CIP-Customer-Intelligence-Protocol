"""Multi-turn conversation with history tracking and context accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cip_protocol.cip import CIP, CIPResult


@dataclass
class Turn:
    user_input: str
    result: CIPResult
    scaffold_id: str
    turn_number: int


class Conversation:
    """Multi-turn wrapper around CIP with history and context accumulation."""

    def __init__(self, cip: CIP, *, max_history_turns: int = 20) -> None:
        self._cip = cip
        self._max_history_turns = max_history_turns
        self._history: list[dict[str, str]] = []
        self._turns: list[Turn] = []
        self._accumulated_context: dict[str, Any] = {}

    async def say(
        self,
        user_input: str,
        *,
        tool_name: str = "",
        data_context: dict[str, Any] | None = None,
        policy: Any = None,
        scaffold_id: str | None = None,
        cross_domain_context: dict[str, Any] | None = None,
    ) -> CIPResult:
        """Send a message and get a response, maintaining conversation state."""
        # Merge accumulated context with new data_context
        merged_context = dict(self._accumulated_context)
        if data_context:
            merged_context.update(data_context)

        result = await self._cip.run(
            user_input,
            tool_name=tool_name,
            data_context=merged_context,
            policy=policy,
            scaffold_id=scaffold_id,
            cross_domain_context=cross_domain_context,
            chat_history=self._history if self._history else None,
        )

        # Append user + assistant to history
        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": result.response.content})

        # Truncate history to max_history_turns * 2 messages (pairs)
        max_messages = self._max_history_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

        # Accumulate context exports
        if result.response.context_exports:
            self._accumulated_context.update(result.response.context_exports)

        turn = Turn(
            user_input=user_input,
            result=result,
            scaffold_id=result.scaffold_id,
            turn_number=len(self._turns) + 1,
        )
        self._turns.append(turn)

        return result

    def reset(self) -> None:
        """Clear conversation state."""
        self._history.clear()
        self._turns.clear()
        self._accumulated_context.clear()

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    @property
    def accumulated_context(self) -> dict[str, Any]:
        return dict(self._accumulated_context)

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def last_scaffold_id(self) -> str | None:
        return self._turns[-1].scaffold_id if self._turns else None
