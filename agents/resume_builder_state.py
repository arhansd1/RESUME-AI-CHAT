from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime, timezone

from pydantic import Field

from pyagenity.state import AgentState
from pyagenity.utils import Message
from pyagenity.utils.constants import START
from pyagenity.state.execution_state import ExecutionState as ExecMeta
from agents.logging_config import logger


class ResumeBuilderState(AgentState):
    """Custom state container for the resume builder graph."""
    jd_summary: Optional[str] = None
    resume_sections: Dict[str, Any] = Field(default_factory=dict)
    section_objects: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    current_section: Optional[str] = None
    src_node: Optional[str] = None
    recommended_answers: Dict[str, List[str]] = Field(default_factory=dict)
    routing_attempts: int = Field(default=0)
    resume_schema: Optional[Dict[str, Any]] = None
    cv_summary: Dict[str, str] = Field(default_factory=dict)


    context: List[Message] = Field(default_factory=list)
    context_summary: Optional[str] = None
    execution_meta: ExecMeta = Field(default_factory=lambda: ExecMeta(current_node=START))

    def make_message(self, role: str, content: str) -> Message:
        msg_dict = {
            "message_id": str(uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            return Message(**msg_dict)
        except Exception as e:
            logger.error(f"Failed to create Message: {e}. Using fallback object.")
            m = Message.__new__(Message)
            for k, v in msg_dict.items():
                setattr(m, k, v)
            return m
