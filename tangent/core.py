# Standard library imports
import copy
import json
from collections import defaultdict
from typing import List, Callable, Union, Optional
import base64
import re
from pathlib import Path
import cv2

# Package/library imports
from openai import OpenAI


# Local imports
from .util import function_to_json, debug_print, merge_chunk, get_instructions
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Structured_Result,
    InstructionsSource,
    _ImageContent,
    _TextContent,
    MessageContent,
)

__CTX_VARS_NAME__ = "extracted_data"

def is_url(text: str) -> bool:
    """Check if text is a URL."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(text) is not None

def is_video(path: str) -> bool:
    """Check if file is a video."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    return Path(path).suffix.lower() in video_extensions

def encode_image(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_video_frames(video_path: str, sample_rate: int = 50) -> List[str]:
    """Extract frames from video and convert to base64."""
    video = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        if frame_count % sample_rate == 0:  # Sample every Nth frame
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        frame_count += 1
    
    video.release()
    return frames

class tangent:
    def __init__(self, client=None):
        if not client:
            client = OpenAI()
        self.client = client

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        extracted_data: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        extracted_data = defaultdict(str, extracted_data)
        instructions = (
            agent.instructions(extracted_data)
            if callable(agent.instructions)
            else agent.instructions
        )
        
        # Process messages for vision
        processed_messages = []
        system_message = {"role": "system", "content": instructions}
        processed_messages.append(system_message)
        
        for msg in history:
            if agent.vision_enabled and msg["role"] == "user":
                content = msg["content"]
                # Handle string content with image
                if isinstance(content, str) and "image" in msg:
                    processed_content = [
                        {"type": "text", "text": content},
                        self._process_vision_input(msg["image"])
                    ]
                # Handle string content with multiple images
                elif isinstance(content, str) and "images" in msg:
                    image_contents = []
                    for img in msg["images"]:
                        result = self._process_vision_input(img)
                        if isinstance(result, list):
                            image_contents.extend(result)
                        else:
                            image_contents.append(result)
                    processed_content = [
                        {"type": "text", "text": content},
                        *image_contents
                    ]
                # Handle pre-formatted vision content
                elif isinstance(content, list):
                    processed_content = content
                else:
                    processed_content = content
                
                processed_messages.append({
                    **msg,
                    "content": processed_content
                })
            else:
                processed_messages.append(msg)
        
        debug_print(debug, "Getting chat completion for...:", processed_messages)

        tools = [function_to_json(f) for f in agent.functions]
        # hide extracted_data from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        create_params = {
            "model": model_override or agent.model,
            "messages": processed_messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": stream,
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        return self.client.chat.completions.create(**create_params)

    def handle_function_result(self, result, debug) -> Structured_Result:
        match result:
            case Structured_Result() as result:
                return result

            case Agent() as agent:
                return Structured_Result(
                    result_overview=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Structured_Result(result_overview=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Structured_Result object. Error: {str(e)}"
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        extracted_data: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(
            messages=[], agent=None, extracted_data={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                debug_print(debug, f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            debug_print(
                debug, f"Processing tool call: {name} with arguments {args}")

            func = function_map[name]
            # pass extracted_data to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = extracted_data
            raw_result = function_map[name](**args)

            result: Structured_Result = self.handle_function_result(raw_result, debug)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.result_overview,
                }
            )
            partial_response.extracted_data.update(result.extracted_data)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        image: Optional[str] = None,
        images: Optional[List[str]] = None,
        extracted_data: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        # Handle image inputs
        if agent.vision_enabled and (image or images):
            processed_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    if image:
                        msg["image"] = image
                    if images:
                        msg["images"] = images
                processed_messages.append(msg)
            messages = processed_messages
            
        active_agent = agent
        extracted_data = copy.deepcopy(extracted_data)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                extracted_data=extracted_data,
                model_override=model_override,
                stream=True,
                debug=debug,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(
                message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating extracted_data, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, extracted_data, debug
            )
            history.extend(partial_response.messages)
            extracted_data.update(partial_response.extracted_data)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                extracted_data=extracted_data,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        image: Optional[str] = None,
        images: Optional[List[str]] = None,
        extracted_data: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                image=image,
                images=images,
                extracted_data=extracted_data,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
            
        # Handle image inputs
        if agent.vision_enabled and (image or images):
            processed_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    if image:
                        msg["image"] = image
                    if images:
                        msg["images"] = images
                processed_messages.append(msg)
            messages = processed_messages
            
        active_agent = agent
        extracted_data = copy.deepcopy(extracted_data)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                extracted_data=extracted_data,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            message = completion.choices[0].message
            debug_print(debug, "Received completion:", message)
            message.sender = active_agent.name
            history.append(
                json.loads(message.model_dump_json())
            )  # to avoid OpenAI types (?)

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating extracted_data, and switching agents
            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent.functions, extracted_data, debug
            )
            history.extend(partial_response.messages)
            extracted_data.update(partial_response.extracted_data)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            extracted_data=extracted_data,
        )

    def _prepare_messages(self, agent: Agent, messages: list, extracted_data: dict = None) -> list:
        """Prepare messages for the API call."""
        # Get the agent's instructions
        instructions = get_instructions(agent)
        
        # Create the system message
        system_message = {
            "role": "system",
            "content": instructions
        }
        
        # Add context variables if provided
        if extracted_data:
            system_message["extracted_data"] = extracted_data
        
        # Return full message list with system message first
        return [system_message] + messages

    def _process_vision_input(self, input_path: str) -> Union[dict, List[dict]]:
        """Convert image input to API format."""
        if is_url(input_path):
            return {
                "type": "image_url",
                "image_url": {
                    "url": input_path,
                    "detail": "high"
                }
            }
        
        # Local file handling
        if is_video(input_path):
            frames = extract_video_frames(input_path)
            return [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame}",
                        "detail": "high"
                    }
                }
                for frame in frames
            ]
        
        # Single image file
        base64_image = encode_image(input_path)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        }