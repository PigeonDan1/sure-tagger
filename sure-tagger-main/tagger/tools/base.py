"""Shared tool result contracts."""

from typing import Any, Dict


TOOL_VERSION = "0.1.0"


class ToolResult:
    """One normalized tool result before final evidence resolution."""

    def __init__(
        self,
        tag_path,
        value,
        tool_name,
        method,
        status="observed",
        confidence=1.0,
        tool_type="deterministic",
        tool_version=TOOL_VERSION,
        evidence=None,
    ):
        self.tag_path = tag_path
        self.value = value
        self.tool_name = tool_name
        self.method = method
        self.status = status
        self.confidence = confidence
        self.tool_type = tool_type
        self.tool_version = tool_version
        self.evidence = evidence or {}

    def to_evidence(self):
        # type: () -> Dict[str, Any]
        return {
            "source": self.tool_name,
            "tool_type": self.tool_type,
            "tool_version": self.tool_version,
            "method": self.method,
            "value": self.value,
            "status": self.status,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }

    def to_record(self):
        # type: () -> Dict[str, Any]
        return {
            "tag_path": self.tag_path,
            "value": self.value,
            "tool_type": self.tool_type,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "method": self.method,
            "status": self.status,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }
