"""
Module: llm_client.py

A lightweight wrapper around OpenAI's synchronous and asynchronous Chat Completion APIs.
Provides methods for streaming and non-streaming responses, decision-style JSON outputs,
token counting, and cost estimation for different GPT models.

Usage:
    from llm_client import GPT
    gpt = GPT(api_key="YOUR_KEY", model="gpt-4o", base_url="https://api.openai.com")
    response = gpt.get_LLM_response("Hello, world!", system_message="You are a helpful assistant.")
"""

from openai import OpenAI, AsyncOpenAI, BadRequestError
import sys
import json
import tiktoken


def _is_response_format_rejection(exc: BaseException) -> bool:
    """True when the API rejected structured output / response_format (retry with looser format)."""
    if not isinstance(exc, BadRequestError):
        return False
    msg = str(exc).lower()
    return "response_format" in msg or "json_schema" in msg


class FatalAPIError(BaseException):
    """Account, billing, or key failure. Subclasses BaseException so per-query handlers cannot continue."""


_FATAL_API_MARKERS = (
    "insufficient balance",
    "insufficient_quota",
    "exceeded your current quota",
    "quota exceeded",
    "payment required",
    "account disabled",
    "account suspended",
    "account deactivated",
    "api key has been disabled",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "余额不足",
    "余额已用尽",
    "欠费",
    "无可用资源",
    "额度不足",
    "已停用",
    "被停用",
    "已冻结",
    "账号异常",
    "账户异常",
    "1113",
)


def _fatal_api_reason(exc: BaseException) -> str | None:
    """Return the provider error text when the account cannot continue, else None."""
    status = getattr(exc, "status_code", None)
    parts = [str(exc)]
    response = getattr(exc, "response", None)
    text = getattr(response, "text", None) if response is not None else None
    if text:
        parts.append(str(text))
    body = "\n".join(part for part in parts if part)
    lowered = body.lower()
    if status in (401, 402, 403):
        return body.strip() or f"HTTP {status}"
    if any(marker.lower() in lowered for marker in _FATAL_API_MARKERS):
        return body.strip()
    return None



class GPT:
    """
    Lightweight LLM client with optional DeepSeek-thinking support and
    simple token/cost accounting.
    """

    # Global accumulators across all GPT instances (for the whole process)
    _global_input_tokens = 0
    _global_output_tokens = 0
    _global_cost_rmb = 0.0
    _global_cache_hit_tokens = 0
    _global_cache_miss_tokens = 0
    _global_cache_prompt_tokens = 0
    _global_billed_output_tokens = 0

    def __init__(
        self,
        api_key,
        model,
        base_url,
        thinking_type: str | None = None,
        reasoning_effort: str | None = None,
    ):
        """
        Initialize the GPT client.

        Args:
            api_key (str): Your OpenAI API key
            model (str): The GPT model to use (e.g., 'gpt-4o', 'gpt-3.5-turbo', 'deepseek-v3.2')
            base_url (str): The base URL for the OpenAI API
        """
        self.api_key = api_key
        self.model = model or ""
        self.base_url = base_url
        self.thinking_type = (thinking_type or "").strip().lower()
        self.reasoning_effort = (reasoning_effort or "").strip().lower()

        # Per-instance accumulators
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_rmb = 0.0

        # Legacy switch for older DeepSeek thinking models
        self.is_deepseek_thinking = "deepseek" in self.model.lower() and ("v3" in self.model or "thinking" in self.model)

        # Initialize synchronous client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Initialize asynchronous client
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _abort_if_fatal_api(self, exc: BaseException) -> None:
        """Stop the process when the provider rejects the call for billing or account status."""
        reason = _fatal_api_reason(exc)
        if reason is None:
            return
        message = (
            "API 调用失败，已终止运行。\n"
            f"模型: {self.model}\n"
            f"接口: {self.base_url}\n"
            f"原因: {reason}"
        )
        print(message, file=sys.stderr, flush=True)
        raise FatalAPIError(message) from exc

    def _count_tokens(self, text: str) -> int:
        """
        Rough token count using a generic tokenizer.
        This is an approximation but good enough for cost estimation.
        """
        if not text:
            return 0
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            # Fallback heuristic: 1 token ≈ 2 characters
            return max(1, len(text) // 2)

    def _rmb_prices_per_token(self):
        """
        RMB per token: (cached input, uncached input, output).

        DeepSeek-V4-Pro uses the off-peak column: 0.15 / 4.5 / 13.5 yuan per million.
        GLM-5.3 uses 2 / 8 / 28 yuan per million.
        """
        name = (self.model or "").lower()
        if "deepseek-v4-pro" in name:
            return (0.15 / 1_000_000, 4.5 / 1_000_000, 13.5 / 1_000_000)
        if "glm-5.3" in name:
            return (2.0 / 1_000_000, 8.0 / 1_000_000, 28.0 / 1_000_000)
        return None

    def _bill_rmb(self, cache_hit_tokens: int, uncached_input_tokens: int, output_tokens: int) -> float:
        prices = self._rmb_prices_per_token()
        if prices is None:
            return 0.0
        hit_price, miss_price, output_price = prices
        return (
            cache_hit_tokens * hit_price
            + uncached_input_tokens * miss_price
            + output_tokens * output_price
        )

    def _update_usage(self, prompt: str, response: str):
        """Keep a local token count. Money is billed from provider usage, not this estimate."""
        in_tokens = self._count_tokens(prompt)
        out_tokens = self._count_tokens(response)
        self.input_tokens += in_tokens
        self.output_tokens += out_tokens
        GPT._global_input_tokens += in_tokens
        GPT._global_output_tokens += out_tokens

    def _wants_prompt_cache(self) -> bool:
        return "claude" in (self.model or "").lower()

    @staticmethod
    def _is_cache_control_rejection(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(
            token in msg
            for token in (
                "cache_control",
                "cache control",
                "prompt caching",
                "expected a string",
                "must be a string",
            )
        )

    @staticmethod
    def _has_cache_control(messages: list) -> bool:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("cache_control"):
                    return True
        return False

    @staticmethod
    def _strip_cache_control(messages: list) -> list:
        stripped = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                text = "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
                stripped.append({"role": message.get("role"), "content": text})
            else:
                stripped.append(message)
        return stripped

    def _compose_messages(self, prompt: str, system_message: str | None) -> list:
        """Stable system prefix first, query-specific user text second."""
        messages = []
        if system_message:
            if self._wants_prompt_cache():
                messages.append({
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": system_message,
                        "cache_control": {"type": "ephemeral"},
                    }],
                })
            else:
                messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _cache_counts_from_usage(usage):
        """Return (cache_hit, uncached_input, output) from a provider usage object, or None."""
        if usage is None:
            return None
        if isinstance(usage, dict):
            data = usage
        else:
            data = {}
            for key in (
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
                "prompt_tokens",
                "completion_tokens",
                "prompt_tokens_details",
            ):
                if hasattr(usage, key):
                    data[key] = getattr(usage, key)
            model_dump = getattr(usage, "model_dump", None)
            if callable(model_dump):
                dumped = model_dump()
                if isinstance(dumped, dict):
                    data.update(dumped)

        output = data.get("completion_tokens")
        output_n = int(output or 0)
        hit = data.get("prompt_cache_hit_tokens")
        miss = data.get("prompt_cache_miss_tokens")
        if hit is not None or miss is not None:
            return int(hit or 0), int(miss or 0), output_n

        details = data.get("prompt_tokens_details")
        cached = None
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
        elif details is not None and hasattr(details, "cached_tokens"):
            cached = getattr(details, "cached_tokens")
        prompt = data.get("prompt_tokens")
        if cached is None or prompt is None:
            if output_n <= 0:
                return None
            return 0, 0, output_n
        hit_n = int(cached or 0)
        miss_n = max(0, int(prompt or 0) - hit_n)
        return hit_n, miss_n, output_n

    def _note_chunk_usage(self, chunk) -> None:
        usage = getattr(chunk, "usage", None)
        counts = self._cache_counts_from_usage(usage)
        if counts is None:
            return
        hit, miss, output = counts
        if hit <= 0 and miss <= 0 and output <= 0:
            return
        GPT._global_cache_hit_tokens += hit
        GPT._global_cache_miss_tokens += miss
        GPT._global_cache_prompt_tokens += hit + miss
        GPT._global_billed_output_tokens += output
        cost = self._bill_rmb(hit, miss, output)
        self.cost_rmb += cost
        GPT._global_cost_rmb += cost

    def _create_sync(self, messages, extra_params, response_format=None):
        def once(msgs, include_usage=True):
            kwargs = {
                "model": self.model,
                "messages": msgs,
                "stream": True,
                **extra_params,
            }
            if include_usage:
                kwargs["stream_options"] = {"include_usage": True}
            if self.thinking_type != "enabled":
                kwargs["temperature"] = 0.0
            if response_format is not None:
                kwargs["response_format"] = response_format
            try:
                return self.client.chat.completions.create(**kwargs)
            except BadRequestError as exc:
                self._abort_if_fatal_api(exc)
                if include_usage and "stream_options" in str(exc).lower():
                    return once(msgs, include_usage=False)
                raise
            except Exception as exc:
                self._abort_if_fatal_api(exc)
                raise

        try:
            return once(messages)
        except BadRequestError as exc:
            self._abort_if_fatal_api(exc)
            if self._has_cache_control(messages) and self._is_cache_control_rejection(exc):
                print("ℹ️ 网关不接受 cache_control，改为普通 system 文本后重试。")
                return once(self._strip_cache_control(messages))
            raise

    async def _create_async(self, messages, extra_params, response_format=None):
        async def once(msgs, include_usage=True):
            kwargs = {
                "model": self.model,
                "messages": msgs,
                "stream": True,
                **extra_params,
            }
            if include_usage:
                kwargs["stream_options"] = {"include_usage": True}
            if self.thinking_type != "enabled":
                kwargs["temperature"] = 0.0
            if response_format is not None:
                kwargs["response_format"] = response_format
            try:
                return await self.async_client.chat.completions.create(**kwargs)
            except BadRequestError as exc:
                self._abort_if_fatal_api(exc)
                if include_usage and "stream_options" in str(exc).lower():
                    return await once(msgs, include_usage=False)
                raise
            except Exception as exc:
                self._abort_if_fatal_api(exc)
                raise

        try:
            return await once(messages)
        except BadRequestError as exc:
            self._abort_if_fatal_api(exc)
            if self._has_cache_control(messages) and self._is_cache_control_rejection(exc):
                print("ℹ️ 网关不接受 cache_control，改为普通 system 文本后重试。")
                return await once(self._strip_cache_control(messages))
            raise

    @classmethod
    def get_global_usage(cls):
        """
        Get aggregated usage across all GPT instances.
        Returns:
            dict: {input_tokens, output_tokens, total_cost_rmb}
        """
        prompt_tokens = cls._global_cache_prompt_tokens
        hit_tokens = cls._global_cache_hit_tokens
        return {
            "input_tokens": cls._global_input_tokens,
            "output_tokens": cls._global_output_tokens,
            "total_cost_rmb": cls._global_cost_rmb,
            "cache_hit_tokens": hit_tokens,
            "uncached_input_tokens": cls._global_cache_miss_tokens,
            "billed_output_tokens": cls._global_billed_output_tokens,
            "cache_prompt_tokens": prompt_tokens,
            "cache_hit_rate": (hit_tokens / prompt_tokens) if prompt_tokens > 0 else None,
        }

    def get_LLM_response(self, prompt, system_message=None, json_format=False, json_schema=None) -> str:
        """
        Get a streaming response from GPT.

        Args:
            prompt (str): The user prompt to send to GPT
            system_message (str, optional): System message to set context. Defaults to None.
            json_format (bool, optional): Whether to request JSON formatted response. Defaults to False.
            json_schema (dict, optional): Strict JSON schema spec. If provided, uses
                response_format=json_schema with strict mode.

        Returns:
            str: The complete response from GPT
        """
        messages = self._compose_messages(prompt, system_message)
        counted_prompt = f"{system_message}\n{prompt}" if system_message else prompt

        # Prepare extra parameters for DeepSeek/OpenAI-compatible thinking controls
        extra_params = {}
        if self.thinking_type in {"enabled", "disabled"}:
            extra_params["extra_body"] = {"thinking": {"type": self.thinking_type}}
        elif self.is_deepseek_thinking:
            extra_params["extra_body"] = {"enable_thinking": True}
        if self.reasoning_effort in {"low", "high", "max"}:
            extra_params["reasoning_effort"] = self.reasoning_effort

        try:
            return self._complete_response(
                messages, counted_prompt, extra_params, json_format, json_schema
            )
        except FatalAPIError:
            raise
        except Exception as exc:
            self._abort_if_fatal_api(exc)
            raise

    def _complete_response(self, messages, counted_prompt, extra_params, json_format, json_schema) -> str:
        if json_format:
            # Request JSON formatted response (prefer strict schema when provided)
            response_format = {"type": "json_object"}
            if isinstance(json_schema, dict) and json_schema:
                if "type" in json_schema and json_schema.get("type") == "json_schema":
                    response_format = json_schema
                else:
                    schema_name = str(json_schema.get("name", "structured_output"))
                    schema_body = (
                        json_schema.get("schema")
                        if isinstance(json_schema.get("schema"), dict)
                        else json_schema
                    )
                    strict = bool(json_schema.get("strict", True))
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": strict,
                            "schema": schema_body,
                        },
                    }
            try:
                completion = self._create_sync(
                    messages,
                    extra_params,
                    response_format=response_format,
                )
            except BadRequestError as exc:
                self._abort_if_fatal_api(exc)
                # Many OpenAI-compatible gateways support json_object but not json_schema;
                # prompt already constrains shape, so fall back to plain JSON object mode.
                if (
                    isinstance(json_schema, dict)
                    and json_schema
                    and response_format.get("type") == "json_schema"
                    and _is_response_format_rejection(exc)
                ):
                    completion = self._create_sync(
                        messages,
                        extra_params,
                        response_format={"type": "json_object"},
                    )
                elif not (isinstance(json_schema, dict) and json_schema):
                    completion = self._create_sync(
                        messages,
                        extra_params,
                        response_format={"type": "json_object"},
                    )
                else:
                    raise
            except Exception as exc:
                self._abort_if_fatal_api(exc)
                if isinstance(json_schema, dict) and json_schema:
                    raise
                completion = self._create_sync(
                    messages,
                    extra_params,
                    response_format={"type": "json_object"},
                )

            if self.is_deepseek_thinking:
                full_response = self._process_deepseek_streaming_response(completion)
            else:
                full_response = ""
                for chunk in completion:
                    self._note_chunk_usage(chunk)
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                        chunk_text = chunk.choices[0].delta.content
                        print(chunk_text, end="", flush=True)
                        full_response += chunk_text
                print()

            # Update cost statistics
            self._update_usage(counted_prompt, full_response)
            return full_response

        else:
            # Request regular text response
            completion = self._create_sync(messages, extra_params)

            if self.is_deepseek_thinking:
                full_response = self._process_deepseek_streaming_response(completion)
            else:
                # Process streaming response
                full_response = ""
                for chunk in completion:
                    self._note_chunk_usage(chunk)
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                        chunk_text = chunk.choices[0].delta.content
                        print(chunk_text, end="", flush=True)
                        full_response += chunk_text
                print()  # New line after response

            # Update cost statistics
            self._update_usage(counted_prompt, full_response)
            return full_response
    
    async def get_LLM_response_async(self, prompt, system_message=None, json_format=False, json_schema=None) -> str:
            """
            Get an asynchronous streaming response from GPT.

            Args:
                prompt (str): The user prompt to send to GPT
                system_message (str, optional): System message to set context. Defaults to None.
                json_format (bool, optional): Whether to request JSON formatted response. Defaults to False.
                json_schema (dict, optional): Strict JSON schema spec. If provided, uses
                    response_format=json_schema with strict mode.

            Returns:
                str: The complete response from GPT
            """
            messages = self._compose_messages(prompt, system_message)
            counted_prompt = f"{system_message}\n{prompt}" if system_message else prompt

            # Prepare extra parameters for DeepSeek/OpenAI-compatible thinking controls
            extra_params = {}
            if self.thinking_type in {"enabled", "disabled"}:
                extra_params["extra_body"] = {"thinking": {"type": self.thinking_type}}
            elif self.is_deepseek_thinking:
                extra_params["extra_body"] = {"enable_thinking": True}
            if self.reasoning_effort in {"low", "high", "max"}:
                extra_params["reasoning_effort"] = self.reasoning_effort

            try:
                return await self._complete_response_async(
                    messages, counted_prompt, extra_params, json_format, json_schema
                )
            except FatalAPIError:
                raise
            except Exception as exc:
                self._abort_if_fatal_api(exc)
                raise

    async def _complete_response_async(self, messages, counted_prompt, extra_params, json_format, json_schema) -> str:
            if json_format:
                # Request JSON formatted response (prefer strict schema when provided)
                response_format = {"type": "json_object"}
                if isinstance(json_schema, dict) and json_schema:
                    if "type" in json_schema and json_schema.get("type") == "json_schema":
                        response_format = json_schema
                    else:
                        schema_name = str(json_schema.get("name", "structured_output"))
                        schema_body = (
                            json_schema.get("schema")
                            if isinstance(json_schema.get("schema"), dict)
                            else json_schema
                        )
                        strict = bool(json_schema.get("strict", True))
                        response_format = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name,
                                "strict": strict,
                                "schema": schema_body,
                            },
                        }
                try:
                    completion = await self._create_async(
                        messages,
                        extra_params,
                        response_format=response_format,
                    )
                except BadRequestError as exc:
                    self._abort_if_fatal_api(exc)
                    if (
                        isinstance(json_schema, dict)
                        and json_schema
                        and response_format.get("type") == "json_schema"
                        and _is_response_format_rejection(exc)
                    ):
                        completion = await self._create_async(
                            messages,
                            extra_params,
                            response_format={"type": "json_object"},
                        )
                    elif not (isinstance(json_schema, dict) and json_schema):
                        completion = await self._create_async(
                            messages,
                            extra_params,
                            response_format={"type": "json_object"},
                        )
                    else:
                        raise
                except Exception as exc:
                    self._abort_if_fatal_api(exc)
                    if isinstance(json_schema, dict) and json_schema:
                        raise
                    completion = await self._create_async(
                        messages,
                        extra_params,
                        response_format={"type": "json_object"},
                    )

                if self.is_deepseek_thinking:
                    full_response = await self._process_deepseek_streaming_response_async(completion)
                else:
                    full_response = ""
                    async for chunk in completion:
                        self._note_chunk_usage(chunk)
                        if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                            chunk_text = chunk.choices[0].delta.content
                            print(chunk_text, end="", flush=True)
                            full_response += chunk_text
                    print()

                # Update cost statistics
                self._update_usage(counted_prompt, full_response)
                return full_response
            else:
                # Request regular text response
                completion = await self._create_async(messages, extra_params)

                if self.is_deepseek_thinking:
                    full_response = await self._process_deepseek_streaming_response_async(completion)
                else:
                    # Process streaming response asynchronously
                    full_response = ""
                    async for chunk in completion:
                        self._note_chunk_usage(chunk)
                        if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                            chunk_text = chunk.choices[0].delta.content
                            print(chunk_text, end="", flush=True)
                            full_response += chunk_text
                    print()  # New line after response

                # Update cost statistics
                self._update_usage(counted_prompt, full_response)
                return full_response

    def _process_deepseek_streaming_response(self, completion) -> str:
        """
        Process DeepSeek streaming response with thinking content.

        Args:
            completion: The streaming completion object

        Returns:
            str: The complete response content (without thinking)
        """
        full_response = ""
        is_answering = False

        for chunk in completion:
            self._note_chunk_usage(chunk)
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            # Handle reasoning content (thinking process)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    # Only print thinking content before the actual answer starts
                    print(delta.reasoning_content, end="", flush=True)

            # Handle actual content
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                    is_answering = True
                print(delta.content, end="", flush=True)
                full_response += delta.content

        print()  # New line after response
        return full_response

    async def _process_deepseek_streaming_response_async(self, completion) -> str:
        """
        Process DeepSeek streaming response asynchronously with thinking content.

        Args:
            completion: The async streaming completion object

        Returns:
            str: The complete response content (without thinking)
        """
        full_response = ""
        is_answering = False

        async for chunk in completion:
            self._note_chunk_usage(chunk)
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            # Handle reasoning content (thinking process)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    # Only print thinking content before the actual answer starts
                    print(delta.reasoning_content, end="", flush=True)

            # Handle actual content
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                    is_answering = True
                print(delta.content, end="", flush=True)
                full_response += delta.content

        print()  # New line after response
        return full_response


    # def calc_token(self, in_text, out_text="") -> int:
    #     """
    #     Calculate the number of tokens for given text using the model's tokenizer.
        
    #     Args:
    #         in_text (str): Input text to tokenize
    #         out_text (str, optional): Output text to tokenize. Defaults to "".
            
    #     Returns:
    #         int: Total number of tokens
    #     """
    #     enc = tiktoken.encoding_for_model(self.model)
    #     return len(enc.encode(str(out_text) + str(in_text)))

    # def calc_money(self, in_text, out_text) -> float:
    #     """
    #     Calculate the estimated cost for API usage based on token count.
        
    #     Args:
    #         in_text (str): Input text
    #         out_text (str): Output text
            
    #     Returns:
    #         float: Estimated cost in USD
            
    #     Note:
    #         Pricing is based on OpenAI's pricing as of the implementation date.
    #         Actual costs may vary. Please refer to OpenAI's current pricing.
    #     """
    #     if self.model == "gpt-4o":
    #         return (self.calc_token(in_text) * 0.005 + self.calc_token(out_text) * 0.015) / 1000
    #     elif self.model == "gpt-4o-mini":
    #         return (self.calc_token(in_text) * 0.00015 + self.calc_token(out_text) * 0.0006) / 1000
    #     elif self.model == "gpt-4o-2024-08-06":
    #         return (self.calc_token(in_text) * 0.0025 + self.calc_token(out_text) * 0.01) / 1000
    #     elif self.model == "gpt-4":
    #         return (self.calc_token(in_text) * 0.03 + self.calc_token(out_text) * 0.06) / 1000
    #     elif self.model == "gpt-3.5-turbo":
    #         return (self.calc_token(in_text) * 0.0015 + self.calc_token(out_text) * 0.002) / 1000



# # Test with system message
# gpt = GPT()

# # Example 1: With a system message
# system_message = "You are a helpful assistant that provides factual answers to general knowledge questions."
# prompt = "What is the capital of France?"
# response = gpt.get_LLM_response(prompt, system_message=system_message, json_format=False)
# print(response)

# # Example 2: Without system message
# response_no_system = gpt.get_LLM_response(prompt, system_message=None, json_format=False)
# print(f"Response without system message: {response_no_system}")



