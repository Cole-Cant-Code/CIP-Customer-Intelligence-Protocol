"""CIP Protocol — Customer Intelligence Protocol framework.

The protocol provides domain-agnostic infrastructure for building
consumer-facing MCP servers.  Domains plug in via DomainConfig.

Public API::

    from cip_protocol import DomainConfig
    from cip_protocol.scaffold import ScaffoldEngine, ScaffoldRegistry, load_scaffold_directory
    from cip_protocol.llm import InnerLLMClient, create_provider
"""

from cip_protocol.domain import DomainConfig

__all__ = ["DomainConfig"]
__version__ = "0.1.0"
