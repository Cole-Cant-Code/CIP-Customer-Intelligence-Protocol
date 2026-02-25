"""Interactive CIP playground REPL."""

from __future__ import annotations

import asyncio
from argparse import Namespace

from cip_protocol.cip import CIP, CIPResult
from cip_protocol.conversation import Conversation
from cip_protocol.domain import DomainConfig


def _make_config(args: Namespace) -> DomainConfig:
    return DomainConfig(
        name=args.domain,
        display_name=f"CIP: {args.domain}",
        system_prompt="You are a helpful assistant. Analyze data and respond clearly.",
        default_scaffold_id=args.default_scaffold if args.default_scaffold else None,
    )


def _print_result(result: CIPResult) -> None:
    mode = result.selection_mode
    score_str = ""
    if result.selection_scores:
        best = result.selection_scores.get(result.scaffold_id, 0.0)
        score_str = f" | score: {best:.2f}"
    print(f"[scaffold: {result.scaffold_id} | mode: {mode}{score_str}]")
    print(f"cip> {result.response.content}")
    print()


def _print_explain(result: CIPResult | None) -> None:
    if not result:
        print("No previous result to explain.")
        return
    print(f"Selection mode: {result.selection_mode}")
    print(f"Selected: {result.scaffold_id} ({result.scaffold_display_name})")
    if result.policy_source:
        print(f"Policy: {result.policy_source}")
    if result.unrecognized_constraints:
        print(f"Unrecognized: {', '.join(result.unrecognized_constraints)}")
    if result.selection_scores:
        print("Scores:")
        for sid, score in sorted(result.selection_scores.items(), key=lambda x: -x[1]):
            print(f"  {sid}: {score:.2f}")
    print()


def _print_help() -> None:
    print("Commands:")
    print("  /policy <text>   Set constraint policy for subsequent messages")
    print("  /policy clear    Clear current policy")
    print("  /policy show     Show current policy")
    print("  /explain         Explain last scaffold selection")
    print("  /scaffolds       List loaded scaffolds")
    print("  /history         Show conversation history")
    print("  /context         Show accumulated context")
    print("  /reset           Reset conversation")
    print("  /help            Show this help")
    print("  /quit            Exit playground")
    print()


class _PlaygroundState:
    def __init__(self, cip: CIP) -> None:
        self.cip = cip
        self.conv: Conversation = cip.conversation()
        self.policy_text: str = ""
        self.last_result: CIPResult | None = None


async def _handle_input(state: _PlaygroundState, line: str) -> bool:
    """Handle one line of input. Returns False to quit."""
    stripped = line.strip()
    if not stripped:
        return True

    # Commands
    if stripped.startswith("/"):
        parts = stripped.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/quit" or cmd == "/exit":
            return False

        if cmd == "/help":
            _print_help()
            return True

        if cmd == "/policy":
            if arg.lower() == "clear" or not arg:
                state.policy_text = ""
                print("Policy cleared.")
            elif arg.lower() == "show":
                print(f"Current policy: {state.policy_text or '(none)'}")
            else:
                state.policy_text = arg
                print(f"Policy set: {state.policy_text}")
            print()
            return True

        if cmd == "/explain":
            _print_explain(state.last_result)
            return True

        if cmd == "/scaffolds":
            scaffolds = state.cip.registry.all()
            if not scaffolds:
                print("No scaffolds loaded.")
            else:
                print(f"Scaffolds loaded: {len(scaffolds)}")
                for s in scaffolds:
                    tools = ", ".join(s.applicability.tools) if s.applicability.tools else "(none)"
                    print(f"  {s.id}: {s.display_name} [tools: {tools}]")
            print()
            return True

        if cmd == "/history":
            history = state.conv.history
            if not history:
                print("No history yet.")
            else:
                for msg in history:
                    role = msg["role"]
                    content = msg["content"]
                    preview = content[:80] + "..." if len(content) > 80 else content
                    print(f"  [{role}] {preview}")
            print()
            return True

        if cmd == "/context":
            ctx = state.conv.accumulated_context
            if not ctx:
                print("No accumulated context.")
            else:
                for k, v in ctx.items():
                    print(f"  {k}: {v}")
            print()
            return True

        if cmd == "/reset":
            state.conv.reset()
            state.last_result = None
            state.policy_text = ""
            print("Conversation reset.")
            print()
            return True

        print(f"Unknown command: {cmd}. Type /help for available commands.")
        print()
        return True

    # Regular message
    policy = state.policy_text if state.policy_text else None
    try:
        result = await state.conv.say(stripped, policy=policy)
    except Exception as exc:
        print(f"Error: {exc}")
        print()
        return True
    state.last_result = result
    _print_result(result)
    return True


def run_playground(args: Namespace) -> None:
    config = _make_config(args)
    cip = CIP.from_config(
        config, args.scaffold_dir, args.provider,
        api_key=args.api_key, model=args.model,
    )

    scaffold_count = len(cip.registry.all())
    print()
    print("CIP Playground")
    print(f"Provider: {args.provider} | Domain: {args.domain}")
    print(f"Scaffolds loaded: {scaffold_count}")
    print("Type /help for commands, /quit to exit.")
    print()

    state = _PlaygroundState(cip)

    async def _loop() -> None:
        while True:
            try:
                line = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not await _handle_input(state, line):
                break

    try:
        asyncio.run(_loop())
    except KeyboardInterrupt:
        pass
    print("Goodbye.")
