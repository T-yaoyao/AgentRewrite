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

from openai import OpenAI, AsyncOpenAI
import sys
import json
import tiktoken



class GPT:
    """
    Lightweight LLM client with optional DeepSeek-thinking support and
    simple token/cost accounting.
    """

    # Global accumulators across all GPT instances (for the whole process)
    _global_input_tokens = 0
    _global_output_tokens = 0
    _global_cost_rmb = 0.0

    def __init__(self, api_key, model, base_url):
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

        # Per-instance accumulators
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_rmb = 0.0

        # Check if this is a DeepSeek thinking model
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

    def _estimate_cost_rmb(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate RMB cost based on model pricing.

        For deepseek-v3.2 (from screenshot):
            - Input:  0.002 RMB / 1K tokens
            - Output: 0.003 RMB / 1K tokens

        For other models, we currently return 0 (can be extended later).
        """
        model_lower = self.model.lower()
        if "deepseek-v3.2" in model_lower or ("deepseek" in model_lower and "v3" in model_lower):
            in_price = 0.002  # RMB per 1K tokens
            out_price = 0.003
        else:
            # Unknown pricing → don't charge
            in_price = 0.0
            out_price = 0.0

        return (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price

    def _update_usage(self, prompt: str, response: str):
        """
        Update per-instance and global token/cost statistics.
        """
        in_tokens = self._count_tokens(prompt)
        out_tokens = self._count_tokens(response)
        cost = self._estimate_cost_rmb(in_tokens, out_tokens)

        # Update instance-level stats
        self.input_tokens += in_tokens
        self.output_tokens += out_tokens
        self.cost_rmb += cost

        # Update global stats
        GPT._global_input_tokens += in_tokens
        GPT._global_output_tokens += out_tokens
        GPT._global_cost_rmb += cost

    @classmethod
    def get_global_usage(cls):
        """
        Get aggregated usage across all GPT instances.
        Returns:
            dict: {input_tokens, output_tokens, total_cost_rmb}
        """
        return {
            "input_tokens": cls._global_input_tokens,
            "output_tokens": cls._global_output_tokens,
            "total_cost_rmb": cls._global_cost_rmb,
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
        # Prepare messages array
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Prepare extra parameters for DeepSeek thinking models
        extra_params = {}
        if self.is_deepseek_thinking:
            extra_params["extra_body"] = {"enable_thinking": True}

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
                completion = self.client.chat.completions.create(
                    temperature=0.0,
                    model=self.model,
                    response_format=response_format,
                    messages=messages,
                    stream=True,
                    **extra_params
                )
            except Exception:
                if isinstance(json_schema, dict) and json_schema:
                    raise
                completion = self.client.chat.completions.create(
                    temperature=0.0,
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=messages,
                    stream=True,
                    **extra_params
                )

            if self.is_deepseek_thinking:
                full_response = self._process_deepseek_streaming_response(completion)
            else:
                full_response = ""
                for chunk in completion:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                        chunk_text = chunk.choices[0].delta.content
                        print(chunk_text, end="", flush=True)
                        full_response += chunk_text
                print()

            # Update cost statistics
            self._update_usage(prompt, full_response)
            return full_response

        else:
            # Request regular text response
            completion = self.client.chat.completions.create(
                temperature=0.0,
                model=self.model,
                messages=messages,
                stream=True,
                **extra_params
            )

            if self.is_deepseek_thinking:
                full_response = self._process_deepseek_streaming_response(completion)
            else:
                # Process streaming response
                full_response = ""
                for chunk in completion:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                        chunk_text = chunk.choices[0].delta.content
                        print(chunk_text, end="", flush=True)
                        full_response += chunk_text
                print()  # New line after response

            # Update cost statistics
            self._update_usage(prompt, full_response)
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
            # Prepare messages array
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            # Prepare extra parameters for DeepSeek thinking models
            extra_params = {}
            if self.is_deepseek_thinking:
                extra_params["extra_body"] = {"enable_thinking": True}

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
                    completion = await self.async_client.chat.completions.create(
                        temperature=0.0,
                        model=self.model,
                        response_format=response_format,
                        messages=messages,
                        stream=True,
                        **extra_params
                    )
                except Exception:
                    if isinstance(json_schema, dict) and json_schema:
                        raise
                    completion = await self.async_client.chat.completions.create(
                        temperature=0.0,
                        model=self.model,
                        response_format={"type": "json_object"},
                        messages=messages,
                        stream=True,
                        **extra_params
                    )

                if self.is_deepseek_thinking:
                    full_response = await self._process_deepseek_streaming_response_async(completion)
                else:
                    full_response = ""
                    async for chunk in completion:
                        if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                            chunk_text = chunk.choices[0].delta.content
                            print(chunk_text, end="", flush=True)
                            full_response += chunk_text
                    print()

                # Update cost statistics
                self._update_usage(prompt, full_response)
                return full_response
            else:
                # Request regular text response
                completion = await self.async_client.chat.completions.create(
                    temperature=0.0,
                    model=self.model,
                    messages=messages,
                    stream=True,
                    **extra_params
                )

                if self.is_deepseek_thinking:
                    full_response = await self._process_deepseek_streaming_response_async(completion)
                else:
                    # Process streaming response asynchronously
                    full_response = ""
                    async for chunk in completion:
                        if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                            chunk_text = chunk.choices[0].delta.content
                            print(chunk_text, end="", flush=True)
                            full_response += chunk_text
                    print()  # New line after response

                # Update cost statistics
                self._update_usage(prompt, full_response)
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



