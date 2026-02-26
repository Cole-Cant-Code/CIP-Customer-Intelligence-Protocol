"""Provider pool — lazy multi-provider CIP instance management."""

from __future__ import annotations

import os

from cip_protocol.cip import CIP
from cip_protocol.domain import DomainConfig

_DEFAULT_KEY_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
}


class ProviderPool:
    """Lazy pool of CIP instances keyed by provider name.

    Encapsulates the provider resolution, model lookup, API-key lookup, and
    CIP construction that domain servers previously duplicated as module globals.

    Parameters
    ----------
    config:
        The ``DomainConfig`` that every CIP instance will use.
    scaffold_dir:
        Path to the scaffold YAML directory.
    key_map:
        ``{provider: ENV_VAR}`` mapping.  Defaults to Anthropic + OpenAI.
    default_models:
        ``{provider: model_id}`` fallback models.
    """

    def __init__(
        self,
        config: DomainConfig,
        scaffold_dir: str,
        *,
        key_map: dict[str, str] | None = None,
        default_models: dict[str, str] | None = None,
    ) -> None:
        self._config = config
        self._scaffold_dir = scaffold_dir
        self._key_map = key_map if key_map is not None else dict(_DEFAULT_KEY_MAP)
        self._default_models = (
            default_models if default_models is not None else dict(_DEFAULT_MODELS)
        )
        self._pool: dict[str, CIP] = {}
        self._provider_models: dict[str, str] = {}
        self._default_provider: str = ""
        self._override: CIP | None = None

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalize(provider: str) -> str:
        return provider.strip().lower()

    def _resolve_provider(self, provider: str = "") -> str:
        resolved = (
            self._normalize(provider)
            or self._normalize(self._default_provider)
            or self._normalize(os.environ.get("CIP_LLM_PROVIDER", "anthropic"))
        )
        if resolved not in self._key_map:
            raise ValueError(
                f"Unknown provider '{resolved}'. "
                f"Use one of: {', '.join(sorted(self._key_map))}."
            )
        return resolved

    def _resolve_model(self, provider: str) -> str:
        model = self._provider_models.get(provider, "").strip()
        if model:
            return model
        env_provider = self._normalize(os.environ.get("CIP_LLM_PROVIDER", "anthropic"))
        env_model = os.environ.get("CIP_LLM_MODEL", "").strip()
        if env_model and provider == env_provider:
            self._provider_models[provider] = env_model
            return env_model
        return ""

    def _build(self, provider: str, model: str = "") -> CIP:
        api_key = os.environ.get(self._key_map.get(provider, ""), "")
        resolved_model = model or self._default_models.get(provider, "")
        return CIP.from_config(
            self._config,
            self._scaffold_dir,
            provider,
            api_key=api_key,
            model=resolved_model,
        )

    # ── public API ────────────────────────────────────────────────

    def get(self, provider: str = "") -> CIP:
        """Return a CIP instance for *provider*, creating one lazily if needed.

        If an override is set (via :meth:`set_override`), it is always returned.
        """
        if self._override is not None:
            return self._override

        resolved = self._resolve_provider(provider)
        if not self._default_provider:
            self._default_provider = resolved

        if resolved not in self._pool:
            model = self._resolve_model(resolved)
            self._pool[resolved] = self._build(resolved, model)

        return self._pool[resolved]

    def set_override(self, cip: CIP | None) -> None:
        """Inject a CIP instance for testing, or ``None`` to clear."""
        self._override = cip

    def set_provider(self, provider: str, model: str = "") -> str:
        """Switch the default provider (and optionally model).  Returns status string."""
        provider = self._normalize(provider)
        if provider not in self._key_map:
            known = ", ".join(sorted(self._key_map))
            return f"Unknown provider '{provider}'. Use one of: {known}."

        resolved_model = (
            model.strip()
            or self._provider_models.get(provider, "").strip()
            or self._default_models.get(provider, "")
        )
        self._provider_models[provider] = resolved_model
        self._default_provider = provider
        self._pool[provider] = self._build(provider, resolved_model)
        return f"CIP reasoning now uses {provider}/{resolved_model}."

    def get_info(self) -> str:
        """Return a human-readable status string."""
        try:
            resolved_default = self._resolve_provider(self._default_provider)
        except ValueError as exc:
            return str(exc)

        if not self._default_provider:
            self._default_provider = resolved_default

        resolved_model = (
            self._provider_models.get(resolved_default, "").strip()
            or self._resolve_model(resolved_default)
            or self._default_models.get(resolved_default, "")
        )
        initialized = sorted(self._pool.keys())
        pool_text = ", ".join(initialized) if initialized else "none"
        return (
            f"{resolved_default}/{resolved_model} "
            f"(default={resolved_default}, pool=[{pool_text}])"
        )

    def prepare_orchestration(
        self,
        *,
        tool_name: str,
        provider: str = "",
        scaffold_id: str = "",
        policy: str = "",
        context_notes: str = "",
    ) -> tuple[CIP, str | None, str | None, str | None]:
        """Resolve provider + normalize orchestration parameters.

        Returns ``(cip, scaffold_id | None, policy | None, context_notes | None)``.
        """
        cip = self.get(provider)
        resolved_scaffold_id = _normalize_optional(scaffold_id)
        if resolved_scaffold_id and cip.registry.get(resolved_scaffold_id) is None:
            raise ValueError(
                f"Unknown scaffold_id '{resolved_scaffold_id}' for tool '{tool_name}'."
            )
        return (
            cip,
            resolved_scaffold_id,
            _normalize_optional(policy),
            _normalize_optional(context_notes),
        )


def _normalize_optional(value: str) -> str | None:
    normalized = value.strip()
    return normalized if normalized else None
