import ast
import asyncio
import importlib
import inspect
import math
import operator
from datetime import datetime
from typing import Any, Callable, Dict


from utils.logger import logger


TOOL_REGISTRY: Dict[str, Callable] = {}

MAX_CALC_EXPRESSION_LENGTH = 500
MAX_CALC_AST_NODES = 100
MAX_POWER_EXPONENT = 10000
MAX_CALC_RESULT_DIGITS = 10000

TOOL_MODULES = (
    "tools.web_search",
    "tools.computer",
    "tools.file_ops",
    "automation.triggers",
)

EXPECTED_TOOLS = {
    "get_current_time",
    "calculate",
    "web_search",
    "browser_search",
    "browser_open_url",
    "fetch_webpage",
    "take_screenshot",
    "click_mouse",
    "type_text",
    "press_key",
    "get_screen_size",
    "read_file",
    "write_file",
    "list_directory",
    "delete_file",
    "schedule_reminder",
}


def register_tool(
    name: str | None = None,
):
    """Decorator used by EDITH tools to register themselves."""

    def decorator(
        func: Callable,
    ):
        tool_name = (
            name.strip()
            if isinstance(name, str)
            and name.strip()
            else func.__name__
        )

        if tool_name in TOOL_REGISTRY:
            logger.warning(
                f"Tool '{tool_name}' is already registered. "
                "Replacing the existing registration."
            )

        TOOL_REGISTRY[tool_name] = func

        logger.debug(
            f"Registered EDITH tool: {tool_name}"
        )

        return func

    return decorator


_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _validate_result_size(
    result: Any,
) -> Any:
    """
    Prevent calculator results from becoming excessively large.
    """

    if isinstance(result, int):

        if result == 0:
            return result

        max_bits = int(
            MAX_CALC_RESULT_DIGITS
            / math.log10(2)
        )

        if result.bit_length() > max_bits:
            raise ValueError(
                "Calculation result is too large."
            )

    elif isinstance(result, float):

        if not math.isfinite(result):
            raise ValueError(
                "Calculation produced a non-finite result."
            )

    return result


def _validate_power_size(
    base: Any,
    exponent: Any,
) -> None:
    """
    Prevent exponentiation from creating an excessively
    large intermediate result.
    """

    if not isinstance(
        base,
        (int, float),
    ):
        raise TypeError(
            "Power base must be numeric."
        )

    if not isinstance(
        exponent,
        (int, float),
    ):
        raise TypeError(
            "Power exponent must be numeric."
        )

    if abs(exponent) > MAX_POWER_EXPONENT:
        raise ValueError(
            "Power exponent is too large."
        )

    if (
        exponent > 0
        and abs(base) > 1
    ):
        estimated_digits = (
            exponent
            * math.log10(
                abs(base)
            )
        )

        if estimated_digits > MAX_CALC_RESULT_DIGITS:
            raise ValueError(
                "Power result would be too large."
            )


def _eval_ast(
    node: ast.AST,
) -> Any:
    """Safely evaluate an arithmetic AST."""

    if isinstance(
        node,
        ast.Num,
    ):
        return node.n

    if (
        isinstance(
            node,
            ast.Constant,
        )
        and isinstance(
            node.value,
            (int, float),
        )
    ):
        return node.value

    if isinstance(
        node,
        ast.BinOp,
    ):
        left = _eval_ast(
            node.left
        )

        right = _eval_ast(
            node.right
        )

        op_type = type(
            node.op
        )

        if op_type not in _SAFE_OPERATORS:
            raise ValueError(
                f"Unsupported binary operator: "
                f"{op_type}"
            )

        if op_type is ast.Pow:
            _validate_power_size(
                left,
                right,
            )

        result = _SAFE_OPERATORS[
            op_type
        ](
            left,
            right,
        )

        return _validate_result_size(
            result
        )

    if isinstance(
        node,
        ast.UnaryOp,
    ):
        operand = _eval_ast(
            node.operand
        )

        op_type = type(
            node.op
        )

        if op_type not in _SAFE_OPERATORS:
            raise ValueError(
                f"Unsupported unary operator: "
                f"{op_type}"
            )

        result = _SAFE_OPERATORS[
            op_type
        ](
            operand
        )

        return _validate_result_size(
            result
        )

    raise TypeError(
        f"Unsupported AST node type: "
        f"{type(node)}"
    )


def _validate_calculation_ast(
    parsed_ast: ast.AST,
) -> None:
    """
    Validate calculator AST complexity before evaluation.
    """

    nodes = list(
        ast.walk(
            parsed_ast
        )
    )

    if len(nodes) > MAX_CALC_AST_NODES:
        raise ValueError(
            "Expression is too complex."
        )

    allowed_nodes = (
        ast.Expression,
        ast.Constant,
        ast.Num,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
    )

    for node in nodes:

        if not isinstance(
            node,
            allowed_nodes,
        ):
            raise ValueError(
                "Expression contains an "
                "unsupported operation."
            )


@register_tool(
    name="get_current_time"
)
def get_current_time() -> str:
    """Return the current local system date and time."""

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


@register_tool(
    name="calculate"
)
def calculate(
    expression: str,
) -> str:
    """Evaluate a mathematical expression securely."""

    if expression is None:
        return (
            "Error evaluating expression: "
            "empty expression"
        )

    try:
        clean_expr = (
            str(expression)
            .replace(
                "^",
                "**",
            )
            .strip()
        )

        if not clean_expr:
            return (
                "Error evaluating expression: "
                "empty expression"
            )

        if len(
            clean_expr
        ) > MAX_CALC_EXPRESSION_LENGTH:
            return (
                "Error evaluating expression: "
                "expression is too long"
            )

        parsed_ast = ast.parse(
            clean_expr,
            mode="eval",
        )

        _validate_calculation_ast(
            parsed_ast
        )

        result = _eval_ast(
            parsed_ast.body
        )

        return str(result)

    except Exception as e:

        logger.warning(
            f"Error evaluating math expression "
            f"'{expression}': {e}"
        )

        return (
            f"Error evaluating expression: "
            f"{str(e)}"
        )


def _load_tool_modules() -> None:
    """
    Import EDITH tool modules so their @register_tool
    decorators execute and populate TOOL_REGISTRY.

    Imports happen after the registry and built-in tools
    have been initialized, avoiding the common circular
    import problem with register_tool.
    """

    for module_name in TOOL_MODULES:

        try:
            importlib.import_module(
                module_name
            )

            logger.info(
                f"Loaded tool module: "
                f"{module_name}"
            )

        except ModuleNotFoundError as e:

            logger.warning(
                f"Tool module '{module_name}' "
                f"could not be loaded: {e}"
            )

        except Exception as e:

            logger.error(
                f"Failed to load tool module "
                f"'{module_name}': {e}",
                exc_info=True,
            )


def _validate_expected_tools() -> None:
    """
    Check tools expected to be registered with EDITH's
    tool registry.
    """

    missing_tools = (
        EXPECTED_TOOLS
        - TOOL_REGISTRY.keys()
    )

    if missing_tools:

        logger.warning(
            "Expected EDITH tools are missing: "
            f"{sorted(missing_tools)}"
        )

    else:

        logger.info(
            "All expected EDITH tools are registered."
        )


def list_registered_tools() -> list[str]:
    """Return registered tool names in sorted order."""

    return sorted(
        TOOL_REGISTRY.keys()
    )


def execute_tool(
    tool_name: str,
    **kwargs: Any,
) -> str:
    """
    Execute a registered tool synchronously.

    Async tools are supported when no event loop is
    currently running in the calling thread.
    """

    if tool_name not in TOOL_REGISTRY:

        logger.error(
            f"Tool '{tool_name}' is not registered."
        )

        return (
            f"Error: Tool '{tool_name}' "
            "not found."
        )

    try:

        logger.info(
            f"Executing tool '{tool_name}' "
            f"with args: {kwargs}"
        )

        func = TOOL_REGISTRY[
            tool_name
        ]

        if asyncio.iscoroutinefunction(
            func
        ):

            try:
                asyncio.get_running_loop()

            except RuntimeError:

                result = asyncio.run(
                    func(
                        **kwargs
                    )
                )

            else:

                return (
                    f"Error executing tool "
                    f"'{tool_name}': async tool "
                    "cannot be executed synchronously "
                    "while an event loop is running."
                )

        else:

            result = func(
                **kwargs
            )

            if inspect.isawaitable(
                result
            ):

                try:
                    asyncio.get_running_loop()

                except RuntimeError:

                    result = asyncio.run(
                        result
                    )

                else:

                    return (
                        f"Error executing tool "
                        f"'{tool_name}': tool returned "
                        "an awaitable while an event loop "
                        "is already running."
                    )

        return str(
            result
        )

    except Exception as e:

        logger.error(
            f"Error running tool "
            f"'{tool_name}': {e}",
            exc_info=True,
        )

        return (
            f"Error executing tool: "
            f"{str(e)}"
        )


async def execute_tool_async(
    tool_name: str,
    **kwargs: Any,
) -> str:
    """
    Execute a registered tool asynchronously.

    Supports both synchronous and asynchronous functions.
    """

    if tool_name not in TOOL_REGISTRY:

        logger.error(
            f"Tool '{tool_name}' is not registered."
        )

        return (
            f"Error: Tool '{tool_name}' "
            "not found."
        )

    try:

        logger.info(
            f"Executing tool '{tool_name}' "
            f"with args: {kwargs}"
        )

        func = TOOL_REGISTRY[
            tool_name
        ]

        if asyncio.iscoroutinefunction(
            func
        ):

            result = await func(
                **kwargs
            )

        else:

            result = await asyncio.to_thread(
                func,
                **kwargs,
            )

            if inspect.isawaitable(
                result
            ):
                result = await result

        return str(
            result
        )

    except Exception as e:

        logger.error(
            f"Error running tool "
            f"'{tool_name}': {e}",
            exc_info=True,
        )

        return (
            f"Error executing tool: "
            f"{str(e)}"
        )


def get_tools_description() -> str:
    """Generate a deterministic description of registered tools."""

    if not TOOL_REGISTRY:
        return "Available Tools:\n- None"

    lines = [
        "Available Tools:"
    ]

    for name in sorted(
        TOOL_REGISTRY.keys()
    ):
        func = TOOL_REGISTRY[
            name
        ]

        try:

            signature = inspect.signature(
                func
            )

        except Exception:

            signature = "(...)"

        description = (
            inspect.getdoc(
                func
            )
            or "No description available."
        )

        lines.append(
            f"- {name}{signature}: "
            f"{description}"
        )

    return "\n".join(
        lines
    )


_load_tool_modules()
_validate_expected_tools()