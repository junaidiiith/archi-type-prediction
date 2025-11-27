import json
import re
import os
from typing import List, Dict, Tuple, Union
from openai import OpenAI
from anthropic import Anthropic
from together import Together
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from pydantic import BaseModel

from ..utils.logging import get_logger
from ..utils.logging import get_prompts_data_dir
from dotenv import load_dotenv

load_dotenv()


logger = get_logger(__name__)
PROMPTS_DATA_DIR = get_prompts_data_dir()
NUM_RETRIES = 3


def get_llm_and_key(provider: str) -> Tuple[str, str]:
    """
    Get LLM API key and model name from environment variables.

    Returns:
        Tuple[str, str]: (api_key, model_name)
    """
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    model_id = os.getenv(f"{provider.upper()}_MODEL_NAME")

    if not api_key:
        raise ValueError(
            f"{provider.upper()}_API_KEY environment variable is required. "
            "Please set it with your API key."
        )

    if not model_id:
        raise ValueError(
            f"{provider.upper()}_MODEL_NAME environment variable is required. "
            "Please set it with your model name."
        )

    return api_key, model_id


class LLMService:
    """Service for LLM-based knowledge graph operations"""

    def __init__(self, provider: str = 'openai', model_name: str = None):
        self.provider = provider
        self.model_name = model_name
        self.api_key = os.getenv(f"{provider.upper()}_API_KEY")
        self.model_id = os.getenv(f"{provider.upper()}_MODEL_NAME")

    def get_openai_response(
        self, 
        messages: List[Dict[str, str]], 
        response_format: BaseModel = None
    ) -> str:
        """Get response from LLM using OpenAI"""
        api_key, model_id = get_llm_and_key('openai')   

        client = OpenAI(api_key=api_key)

        if response_format:
            response = client.responses.parse(
                model=model_id,
                input=messages,
                text_format=response_format,
            )
            return response.output_parsed
        else:
            response = client.chat.completions.create(model=model_id, messages=messages)
            return response.choices[0].message.content

    @staticmethod
    def get_anthropic_response(
        self, messages: List[Dict[str, str]], 
        response_format: BaseModel = None
    ) -> str:
        """Get response from LLM using Anthropic"""
        api_key, model_id = get_llm_and_key("anthropic")
        client = Anthropic(api_key=api_key)

        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Anthropic doesn't support system messages, prepend to first user message
                continue
            anthropic_messages.append(msg)

        # Add system message to first user message if exists
        if messages and messages[0]["role"] == "system":
            if len(anthropic_messages) > 0 and anthropic_messages[0]["role"] == "user":
                anthropic_messages[0]["content"] = (
                    f"{messages[0]['content']}\n\n{anthropic_messages[0]['content']}"
                )

        response = client.messages.create(
            model=model_id, max_tokens=1024, messages=anthropic_messages
        )

        return response.content[0].text

    def get_gemini_response(
        self, 
        messages: List[Dict[str, str]], 
        response_format: BaseModel = None
    ) -> str:
        """Get response from LLM using Gemini"""
        api_key, model_id = get_llm_and_key("gemini")

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "system":
                # Add system message as user message
                gemini_messages.append(
                    {"role": "user", "parts": [{"text": msg["content"]}]}
                )
            else:
                gemini_messages.append(
                    {"role": msg["role"], "parts": [{"text": msg["content"]}]}
                )

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_id,
            contents=gemini_messages,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": response_format.model_json_schema(),
            },
        )
        result = response_format.model_validate_json(response.text)
        return result

    @staticmethod
    def get_togetherai_response(
        self, 
        messages: List[Dict[str, str]], 
        response_format: BaseModel = None
    ) -> str:
        """Get response from LLM using TogetherAI"""
        api_key, model_id = get_llm_and_key("togetherai")

        if response_format:
            response = Together(api_key=api_key).chat.completions.create(
                model=model_id,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "schema": response_format.model_json_schema(),
                },
            )
            return response.choices[0].message.content
        else:
            response = Together(api_key=api_key).chat.completions.create(
                model=model_id, messages=messages
            )
            return response.choices[0].message.content

    def get_deepseek_response(
        self, 
        messages: List[Dict[str, str]], 
        response_format: BaseModel = None
    ) -> str:
        """Get response from LLM using DeepSeek"""
        api_key, model_id = get_llm_and_key("deepseek")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        if response_format:
            messages[-1]["content"] += (
                f"\nProvide the output in JSON format as per the following schema:\n{response_format.model_json_schema()}\n"
            )
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        else:
            response = client.chat.completions.create(model=model_id, messages=messages)
            return response.choices[0].message.content

    def get_llm_response(
        self, 
        messages: List[Dict[str, str]],
        response_format: BaseModel = None,
        function_name: str = None,
    ) -> str:
        """Get response from LLM based on configured type"""

        logger.info("Getting response from LLM")
        logger.info(f"Messages: {messages}")

        def get_response():
            logger.info(f"Getting response from {self.provider}")
            if not hasattr(self, f"get_{self.provider.lower()}_response"):
                raise ValueError(
                    f"Invalid LLM provider: {self.provider}. "
                    "Available providers: openai, anthropic, gemini, togetherai, deepseek"
                )
            fn = getattr(self, f"get_{self.provider.lower()}_response")
            num_retries = 0
            while num_retries < NUM_RETRIES:
                try:
                    response = fn(messages, response_format)
                    if isinstance(response, BaseModel):
                        return response.model_dump()
                    return LLMService.parse_json_response(response)
                except Exception as e:
                    num_retries += 1

            return None

        def save_response(response: Union[str, dict], response_file: str):
            if isinstance(response, str):
                with open(response_file, "w") as f:
                    f.write(response)
            else:
                with open(response_file, "w") as f:
                    json.dump(response, f, indent=4)

        def load_response(response_file: str):
            print(f"Loading response from {response_file}")
            with open(response_file, "r") as f:
                data = f.read()
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data

        def save_prompt(content: str, prompt_file: str):
            with open(prompt_file, "w") as f:
                f.write(content)

        if function_name:

            prompts_dir = os.path.dirname(function_name) if os.sep in function_name else PROMPTS_DATA_DIR
            os.makedirs(prompts_dir, exist_ok=True)
            file_name = os.path.basename(function_name)
                
            response_file = os.path.join(prompts_dir, f"{file_name}_response.json")
            if os.path.exists(response_file):
                response = load_response(response_file)
            else:
                prompt_file = os.path.join(prompts_dir, f"{file_name}_prompt.txt")
                content = "\n".join([msg["content"] for msg in messages]) + "\n"
                save_prompt(content, prompt_file)
                logger.info(f"Prompt: {content}")
                response = get_response()
                logger.info(f"Response: {response}")
                save_response(response, response_file)
                logger.info(f"Response saved to {response_file}")
            return response
        else:
            logger.info("Getting response from LLM")
            logger.info(f"Messages: {messages}")
            return get_response()

    def get_llm_response_parallel(
        self,
        messages_list: List[List[Dict[str, str]]],
        response_format: BaseModel = None,
        max_workers: int = 5,
        function_name: str = None,
    ) -> List[Union[str, dict]]:
        """Get response from LLM in parallel, preserving order."""
        
        results = [None] * len(messages_list)  # preallocate in correct order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.get_llm_response,
                    messages=messages_list[i],
                    response_format=response_format,
                    function_name=f"{function_name}_{i}" if function_name else None,
                ): i
                for i in range(len(messages_list))
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {len(messages_list)} LLM responses",
            ):
                idx = futures[future]
                results[idx] = future.result()

        # filter out None if needed
        return results


    @staticmethod
    def parse_json_response(response: str) -> Union[str, dict]:
        """Parse JSON response from LLM, handling common formatting issues."""
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*$", "", response)
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            array_match = re.search(r"\[.*\]", response, re.DOTALL)
            if array_match:
                try:
                    return json.loads(array_match.group())
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(f"Failed to parse JSON response: {response}", response, e.lineno)
            return response
