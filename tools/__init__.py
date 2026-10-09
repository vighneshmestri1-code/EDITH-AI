from tools.registry import TOOL_REGISTRY, execute_tool, get_tools_description
import tools.computer    # V4
import tools.file_ops    # V5
import tools.web_search  # V6
import automation.triggers # V8

__all__ = ["TOOL_REGISTRY", "execute_tool", "get_tools_description"]