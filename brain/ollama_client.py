import asyncio
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import ollama

from config.settings import settings
from utils.logger import logger


class OllamaClient:
    """
    Async client for communicating with the local Ollama server.

    EDITH uses its own text-based JSON tool-calling layer, so
    native Ollama tool calling remains optional.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self.model_name = (
            model_name or settings.OLLAMA_MODEL
        )

        self.host = (
            host or settings.OLLAMA_HOST
        ).rstrip("/")

        self._async_client = ollama.AsyncClient(
            host=self.host
        )

        cpu_count = os.cpu_count() or 8

        self.cpu_threads = max(
            4,
            min(
                cpu_count - 2,
                12,
            ),
        )

        self.options = {
            "num_gpu": getattr(
                settings,
                "OLLAMA_NUM_GPU",
                -1,
            ),
            "num_ctx": getattr(
                settings,
                "OLLAMA_NUM_CTX",
                4096,
            ),
            "temperature": getattr(
                settings,
                "OLLAMA_TEMPERATURE",
                0.3,
            ),
            "num_thread": getattr(
                settings,
                "OLLAMA_NUM_THREAD",
                self.cpu_threads,
            ),
            "num_predict": getattr(
                settings,
                "OLLAMA_NUM_PREDICT",
                384,
            ),
        }

        self.keep_alive = getattr(
            settings,
            "OLLAMA_KEEP_ALIVE",
            "30m",
        )

        logger.info(
            "OllamaClient initialized: "
            f"model={self.model_name}, "
            f"host={self.host}, "
            f"threads={self.options['num_thread']}, "
            f"context={self.options['num_ctx']}, "
            f"keep_alive={self.keep_alive}"
        )

    async def close(self) -> None:
        """
        Close the underlying Ollama HTTP client.
        """

        try:
            await self._async_client.close()

        except Exception as e:
            logger.warning(
                f"Failed to close Ollama client cleanly: {e}"
            )

    async def is_alive_async(self) -> bool:
        """
        Check whether the local Ollama server is reachable.
        """

        try:
            async with httpx.AsyncClient(
                timeout=3.0
            ) as client:

                response = await client.get(
                    f"{self.host}/api/tags"
                )

                if response.status_code == 200:
                    return True

                logger.warning(
                    "Ollama health check returned "
                    f"HTTP {response.status_code}."
                )

                return False

        except Exception as e:
            logger.error(
                "Failed to reach Ollama server "
                f"at {self.host}: {e}"
            )

            return False

    def is_alive(self) -> bool:
        """
        Synchronous fallback for Ollama availability checks.

        Do not call this from an already-running event loop.
        """

        try:
            return asyncio.run(
                self.is_alive_async()
            )

        except Exception as e:
            logger.error(
                f"Synchronous Ollama health check failed: {e}"
            )

            return False

    async def is_model_available_async(self) -> bool:
        """
        Check whether the configured local model is installed.
        """

        try:
            response = await self._async_client.list()

            models = getattr(
                response,
                "models",
                [],
            )

            target = (
                self.model_name
                .lower()
                .strip()
            )

            if not target:
                return False

            target_base = target.split(":")[0]

            for model in models:

                model_name = (
                    getattr(
                        model,
                        "model",
                        "",
                    )
                    or getattr(
                        model,
                        "name",
                        "",
                    )
                    or ""
                )

                model_name = str(
                    model_name
                ).lower().strip()

                if (
                    model_name == target
                    or model_name.startswith(
                        target_base + ":"
                    )
                ):
                    return True

            logger.warning(
                "Configured Ollama model "
                f"'{self.model_name}' is not installed."
            )

            return False

        except Exception as e:
            logger.error(
                "Error checking local model availability: "
                f"{e}"
            )

            return False

    @staticmethod
    def _extract_message_content(
        response: Any,
    ) -> Optional[str]:
        """
        Extract message content from an Ollama response
        object or dictionary.
        """

        if response is None:
            return None

        message = getattr(
            response,
            "message",
            None,
        )

        if message is not None:

            content = getattr(
                message,
                "content",
                None,
            )

            if content is not None:
                return str(content)

        if isinstance(
            response,
            dict,
        ):

            message_data = response.get(
                "message",
                {},
            )

            if isinstance(
                message_data,
                dict,
            ):

                content = message_data.get(
                    "content"
                )

                if content is not None:
                    return str(content)

        return None

    def _build_request_kwargs(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Build a consistent Ollama chat request.
        """

        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "options": self.options,
            "keep_alive": self.keep_alive,
            "stream": stream,
        }

        if tools:
            request_kwargs["tools"] = tools

        return request_kwargs

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send an asynchronous non-streaming chat request.
        """

        if not messages:
            logger.warning(
                "Ollama chat called with empty messages."
            )

            return None

        try:
            request_kwargs = (
                self._build_request_kwargs(
                    messages=messages,
                    stream=False,
                    tools=tools,
                )
            )

            response = (
                await self._async_client.chat(
                    **request_kwargs
                )
            )

            content = (
                self._extract_message_content(
                    response
                )
            )

            if content is None:
                logger.warning(
                    "Ollama returned a response "
                    "without message content."
                )

                return None

            return {
                "role": "assistant",
                "content": content,
            }

        except Exception as e:
            logger.error(
                "Error calling Ollama chat endpoint: "
                f"{e}"
            )

            return None

    async def generate_response(
        self,
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Generate a complete response.
        """

        response = await self.chat(
            messages
        )

        if not response:
            return None

        content = response.get(
            "content"
        )

        if content is None:
            return None

        return str(
            content
        ).strip()

    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[
            List[Dict[str, Any]]
        ] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response chunks from Ollama.

        Errors are logged and terminate the stream rather than
        inserting diagnostic text into EDITH's answer.
        """

        if not messages:
            logger.warning(
                "Ollama streaming called with empty messages."
            )

            return

        try:
            request_kwargs = (
                self._build_request_kwargs(
                    messages=messages,
                    stream=True,
                    tools=tools,
                )
            )

            response_stream = (
                await self._async_client.chat(
                    **request_kwargs
                )
            )

            async for chunk in response_stream:

                content = (
                    self._extract_message_content(
                        chunk
                    )
                )

                if content:
                    yield content

        except Exception as e:
            logger.error(
                "Error in Ollama streaming endpoint: "
                f"{e}"
            )

    async def warm_up(self) -> bool:
        """
        Load the configured model into Ollama before the first
        real user request.

        This reduces first-response latency.
        """

        try:
            logger.info(
                f"Warming up Ollama model: "
                f"{self.model_name}"
            )

            response = await self._async_client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "ping",
                    }
                ],
                options={
                    **self.options,
                    "num_predict": 1,
                },
                keep_alive=self.keep_alive,
                stream=False,
            )

            content = (
                self._extract_message_content(
                    response
                )
            )

            if content is not None:
                logger.info(
                    "Ollama model warm-up completed."
                )

                return True

            logger.warning(
                "Ollama warm-up returned no content."
            )

            return False

        except Exception as e:
            logger.warning(
                f"Ollama warm-up failed: {e}"
            )

            return False