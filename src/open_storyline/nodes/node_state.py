from dataclasses import dataclass

from open_storyline.mcp.sampling_requester import LLMClient
from open_storyline.nodes.node_summary import NodeSummary

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

@dataclass
class NodeState:
    """Node execution state"""
    session_id: str
    artifact_id: str
    lang: str
    node_summary: NodeSummary
    llm: LLMClient
    mcp_ctx: Context[ServerSession, object]
