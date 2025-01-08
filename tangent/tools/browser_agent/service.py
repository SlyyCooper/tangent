from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import textwrap
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import RateLimitError
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError

from .message_manager.service import MessageManager
from .prompts import AgentMessagePrompt, SystemPrompt
from .views import (
	ActionResult,
	AgentError,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
	AgentStepInfo,
)
from tangent.wrappers.browser.browser import Browser
from tangent.wrappers.browser.context import BrowserContext
from tangent.wrappers.browser.views import BrowserState, BrowserStateHistory
from tangent.wrappers.controller.registry.views import ActionModel
from tangent.wrappers.controller.service import Controller
from tangent.wrappers.dom.history_tree_processor.service import (
	DOMHistoryElement,
	HistoryTreeProcessor,
)
from tangent.wrappers.telemetry.service import ProductTelemetry
from tangent.wrappers.telemetry.views import (
	AgentEndTelemetryEvent,
	AgentRunTelemetryEvent,
	AgentStepErrorTelemetryEvent,
)
from tangent.wrappers.utils import time_execution_async
# Added Tangent imports
from tangent import tangent, Agent as TangentAgent, Response

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class Agent:
	def __init__(
		self,
		task: str,
		llm: tangent,
		browser: Browser | None = None,
		browser_context: BrowserContext | None = None,
		controller: Controller = Controller(),
		save_conversation_path: Optional[str] = None,
		max_failures: int = 3,
		retry_delay: int = 10,
		system_prompt_class: Type[SystemPrompt] = SystemPrompt,
		max_input_tokens: int = 128000,
		validate_output: bool = False,
		generate_gif: bool = True,
		include_attributes: list[str] = [
			'title',
			'type',
			'name',
			'role',
			'tabindex',
			'aria-label',
			'placeholder',
			'value',
			'alt',
			'aria-expanded',
		],
		max_error_length: int = 400,
		max_actions_per_step: int = 10,
		tool_call_in_content: bool = True,
		conversation_id: Optional[str] = None,
		debug: bool = False,
	):
		self.agent_id = str(uuid.uuid4())  # unique identifier for the agent
		self.task = task
		self.llm = llm
		self.save_conversation_path = save_conversation_path
		self._last_result = None
		self.include_attributes = include_attributes
		self.max_error_length = max_error_length
		self.generate_gif = generate_gif
		
		# Store conversation ID
		self.conversation_id = conversation_id or str(uuid.uuid4())
		
		# Controller setup
		self.controller = controller
		self.max_actions_per_step = max_actions_per_step

		# Browser setup
		self.injected_browser = browser is not None
		self.injected_browser_context = browser_context is not None

		# Initialize browser first if needed
		self.browser = browser if browser is not None else (None if browser_context else Browser())

		# Initialize browser context
		if browser_context:
			self.browser_context = browser_context
		elif self.browser:
			self.browser_context = BrowserContext(
				browser=self.browser, config=self.browser.config.new_context_config
			)
		else:
			# If neither is provided, create both new
			self.browser = Browser()
			self.browser_context = BrowserContext(browser=self.browser)

		self.system_prompt_class = system_prompt_class

		# Telemetry setup
		self.telemetry = ProductTelemetry()

		# Action and output models setup
		self._setup_action_models()

		self.max_input_tokens = max_input_tokens

		self.message_manager = MessageManager(
			llm=self.llm,
			task=self.task,
			action_descriptions=self.controller.registry.get_prompt_description(),
			system_prompt_class=self.system_prompt_class,
			max_input_tokens=self.max_input_tokens,
			include_attributes=self.include_attributes,
			max_error_length=self.max_error_length,
			max_actions_per_step=self.max_actions_per_step,
			tool_call_in_content=tool_call_in_content,
			debug=debug,
			conversation_id=self.conversation_id,
		)

		# Tracking variables
		self.history: AgentHistoryList = AgentHistoryList(history=[])
		self.n_steps = 1
		self.consecutive_failures = 0
		self.max_failures = max_failures
		self.retry_delay = retry_delay
		self.validate_output = validate_output

		if save_conversation_path:
			logger.info(f'Saving conversation to {save_conversation_path}')

	def _setup_action_models(self) -> None:
		"""Setup dynamic action models from controller's registry.
		
		This method initializes the action models used by the browser agent,
		creating a dynamic AgentOutput type that combines Tangent's structured
		output format with browser-specific actions.
		"""
		# Get the dynamic action model from controller's registry
		self.ActionModel = self.controller.registry.create_action_model()
		# Create output model with the dynamic actions
		self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

	@time_execution_async('--step')
	async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
		"""Execute one step of the task with visual context always enabled"""
		logger.info(f'\n📍 Step {self.n_steps}')
		state = None
		model_output = None
		result: list[ActionResult] = []

		try:
			# Get state with vision always enabled
			state = await self.browser_context.get_state()
			self.message_manager.add_state_message(state, self._last_result, step_info)
			input_messages = self.message_manager.get_messages()
			try:
				model_output = await self.get_next_action(input_messages)
				self._save_conversation(input_messages, model_output)
				self.message_manager._remove_last_state_message()
				self.message_manager.add_model_output(model_output)
				
				# Save conversation state after successful model output
				await self.message_manager.save_conversation()
				
			except Exception as e:
				self.message_manager._remove_last_state_message()
				raise e

			result: list[ActionResult] = await self.controller.multi_act(
				model_output.action, self.browser_context
			)
			self._last_result = result

			if len(result) > 0 and result[-1].is_done:
				logger.info(f'📄 Result: {result[-1].extracted_content}')

			self.consecutive_failures = 0

		except Exception as e:
			result = self._handle_step_error(e)
			self._last_result = result

		finally:
			if not result:
				return
			for r in result:
				if r.error:
					self.telemetry.capture(
						AgentStepErrorTelemetryEvent(
							agent_id=self.agent_id,
							error=r.error,
						)
					)
			if state:
				self._make_history_item(model_output, state, result)

	def _handle_step_error(self, error: Exception) -> list[ActionResult]:
		"""Handle all types of errors that can occur during a step"""
		include_trace = logger.isEnabledFor(logging.DEBUG)
		error_msg = AgentError.format_error(error, include_trace=include_trace)
		prefix = f'❌ Result failed {self.consecutive_failures + 1}/{self.max_failures} times:\n '

		if isinstance(error, (ValidationError, ValueError)):
			logger.error(f'{prefix}{error_msg}')
			if 'Max token limit reached' in error_msg:
				# cut tokens from history
				self.message_manager.max_input_tokens = self.max_input_tokens - 500
				logger.info(
					f'Cutting tokens from history - new max input tokens: {self.message_manager.max_input_tokens}'
				)
				self.message_manager.cut_messages()
			elif 'Could not parse response' in error_msg:
				# give model a hint how output should look like
				error_msg += '\n\nReturn a valid JSON object with the required fields.'

			self.consecutive_failures += 1
		elif isinstance(error, RateLimitError):
			logger.warning(f'{prefix}{error_msg}')
			time.sleep(self.retry_delay)
			self.consecutive_failures += 1
		else:
			logger.error(f'{prefix}{error_msg}')
			self.consecutive_failures += 1

		return [ActionResult(error=error_msg, include_in_memory=True)]

	def _make_history_item(
		self,
		model_output: AgentOutput | None,
		state: BrowserState,
		result: list[ActionResult],
	) -> None:
		"""Create and store a history item using Tangent's structured output format.
		
		This method creates a history record of an agent step, converting action results
		into Tangent's structured format while preserving browser-specific state information.
		
		Args:
			model_output: The agent's output in Tangent's structured format
			state: The current browser state
			result: List of action results to be converted to structured format
		"""
		if model_output:
			interacted_elements = AgentHistory.get_interacted_element(
				model_output, state.selector_map
			)
		else:
			interacted_elements = [None]

		state_history = BrowserStateHistory(
			url=state.url,
			title=state.title,
			tabs=state.tabs,
			interacted_element=interacted_elements,
			screenshot=state.screenshot,
		)

		# Use structured output format
		history_item = AgentHistory(
			model_output=model_output,
			result=[r.to_structured_result() for r in result],
			state=state_history
		)

		self.history.history.append(history_item)

	def _save_conversation(self, input_messages: list[dict], model_output: AgentOutput) -> None:
		"""Save the conversation to a file"""
		if not self.save_conversation_path:
			return

		try:
			with open(self.save_conversation_path, 'a') as f:
				f.write(f'\n\n--- Step {self.n_steps} ---\n')
				f.write('Input Messages:\n')
				for msg in input_messages:
					f.write(f'  {msg["role"]}: {msg["content"]}\n')
				f.write('Model Output:\n')
				f.write(f'  {model_output.model_dump_json(indent=2)}\n')
		except Exception as e:
			logger.error(f'Could not save conversation: {e}')

	async def rerun_history(
		self, history: AgentHistoryList, start_step: int = 1, end_step: Optional[int] = None
	) -> list[ActionResult]:
		"""Rerun the history from a given step"""
		if not history.history:
			logger.warning('No history to rerun')
			return []

		if end_step is None:
			end_step = len(history.history)

		results = []
		for i, item in enumerate(history.history, 1):
			if i < start_step:
				continue
			if i > end_step:
				break

			logger.info(f'\n🔄 Rerunning step {i}')
			self.n_steps = i
			self.history = AgentHistoryList(history=history.history[:i])
			self.message_manager.clear_messages()

			try:
				state = await self.browser_context.get_state(use_vision=self.use_vision)
				self.message_manager.add_state_message(
					state, self._last_result, AgentStepInfo(rerun=True)
				)
				input_messages = self.message_manager.get_messages()
				try:
					model_output = await self.get_next_action(input_messages)
					self._save_conversation(input_messages, model_output)
					self.message_manager._remove_last_state_message()
					self.message_manager.add_model_output(model_output)
				except Exception as e:
					self.message_manager._remove_last_state_message()
					raise e

				result: list[ActionResult] = await self.controller.multi_act(
					model_output.action, self.browser_context
				)
				self._last_result = result
				results.extend(result)

				if len(result) > 0 and result[-1].is_done:
					logger.info(f'📄 Result: {result[-1].extracted_content}')

			except Exception as e:
				result = self._handle_step_error(e)
				self._last_result = result
				results.extend(result)

			finally:
				if state:
					self._make_history_item(model_output, state, result)

		return results

	def _update_action_index(
		self, action: ActionModel, current_state: BrowserState, historical_element: DOMHistoryElement
	) -> Optional[ActionModel]:
		"""
		Update the action index if the element has moved in the DOM.

		Args:
			action: The action to update
			current_state: The current browser state
			historical_element: The historical element from the history

		Returns:
			The updated action or None if the element is not found
		"""
		if not historical_element or not current_state.element_tree:
			return action

		current_element = HistoryTreeProcessor.find_history_element_in_tree(
			historical_element, current_state.element_tree
		)

		if not current_element or current_element.highlight_index is None:
			return None

		old_index = action.get_index()
		if old_index != current_element.highlight_index:
			action.set_index(current_element.highlight_index)
			logger.info(
				f'Element moved in DOM, updated index from {old_index} to {current_element.highlight_index}'
			)

		return action

	async def load_and_rerun(
		self, history_file: Optional[str | Path] = None, **kwargs
	) -> list[ActionResult]:
		"""
		Load history from file and rerun it.

		Args:
		        history_file: Path to the history file
		        **kwargs: Additional arguments passed to rerun_history
		"""
		if not history_file:
			history_file = 'AgentHistory.json'
		history = AgentHistoryList.load_from_file(history_file, self.AgentOutput)
		return await self.rerun_history(history, **kwargs)

	def save_history(self, file_path: Optional[str | Path] = None) -> None:
		"""Save the history to a file"""
		if not file_path:
			file_path = 'AgentHistory.json'
		self.history.save_to_file(file_path)

	def create_history_gif(
		self,
		output_path: str = 'agent_history.gif',
		duration: int = 3000,
		show_goals: bool = True,
		show_task: bool = True,
		show_logo: bool = False,
		font_size: int = 40,
		title_font_size: int = 56,
		goal_font_size: int = 44,
		margin: int = 40,
		line_spacing: float = 1.5,
	) -> None:
		"""Create a GIF from the agent's history with overlaid task and goal text."""
		if not self.history.history:
			logger.warning('No history to create GIF from')
			return

		images = []
		# if history is empty or first screenshot is None, we can't create a gif
		if not self.history.history or not self.history.history[0].state.screenshot:
			logger.warning('No history or first screenshot to create GIF from')
			return

		# Try to load nicer fonts
		try:
			# Try different font options in order of preference
			font_options = ['Helvetica', 'Arial', 'DejaVuSans', 'Verdana']
			font_loaded = False

			for font_name in font_options:
				try:
					regular_font = ImageFont.truetype(font_name, font_size)
					title_font = ImageFont.truetype(font_name, title_font_size)
					goal_font = ImageFont.truetype(font_name, goal_font_size)
					font_loaded = True
					break
				except OSError:
					continue

			if not font_loaded:
				raise OSError('No preferred fonts found')

		except OSError:
			regular_font = ImageFont.load_default()
			title_font = ImageFont.load_default()

			goal_font = regular_font

		# Load logo if requested
		logo = None
		if show_logo:
			try:
				logo = Image.open('./static/browser-use.png')
				# Resize logo to be small (e.g., 40px height)
				logo_height = 150
				aspect_ratio = logo.width / logo.height
				logo_width = int(logo_height * aspect_ratio)
				logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
			except Exception as e:
				logger.warning(f'Could not load logo: {e}')

		# Create task frame if requested
		if show_task and self.task:
			task_frame = self._create_task_frame(
				self.task,
				self.history.history[0].state.screenshot,
				title_font,
				regular_font,
				logo,
				line_spacing,
			)
			images.append(task_frame)

		# Process each history item
		for i, item in enumerate(self.history.history, 1):
			if not item.state.screenshot:
				continue

			# Convert base64 screenshot to PIL Image
			img_data = base64.b64decode(item.state.screenshot)
			image = Image.open(io.BytesIO(img_data))

			if show_goals and item.model_output:
				image = self._add_overlay_to_image(
					image=image,
					step_number=i,
					goal_text=item.model_output.current_state.next_goal,
					regular_font=regular_font,
					title_font=title_font,
					margin=margin,
					logo=logo,
				)

			images.append(image)

		if images:
			# Save the GIF
			images[0].save(
				output_path,
				save_all=True,
				append_images=images[1:],
				duration=duration,
				loop=0,
				optimize=False,
			)
			logger.info(f'Created GIF at {output_path}')
		else:
			logger.warning('No images found in history to create GIF')

	def _create_task_frame(
		self,
		task: str,
		first_screenshot: str,
		title_font: ImageFont.FreeTypeFont,
		regular_font: ImageFont.FreeTypeFont,
		logo: Optional[Image.Image] = None,
		line_spacing: float = 1.5,
	) -> Image.Image:
		"""Create initial frame showing the task."""
		img_data = base64.b64decode(first_screenshot)
		template = Image.open(io.BytesIO(img_data))
		image = Image.new('RGB', template.size, (0, 0, 0))
		draw = ImageDraw.Draw(image)

		# Calculate vertical center of image
		center_y = image.height // 2

		# Draw task text with increased font size
		margin = 140  # Increased margin
		max_width = image.width - (2 * margin)
		larger_font = ImageFont.truetype(
			regular_font.path, regular_font.size + 16
		)  # Increase font size more
		wrapped_text = self._wrap_text(task, larger_font, max_width)

		# Calculate line height with spacing
		line_height = larger_font.size * line_spacing

		# Split text into lines and draw with custom spacing
		lines = wrapped_text.split('\n')
		total_height = line_height * len(lines)

		# Start position for first line
		text_y = center_y - (total_height / 2) + 50  # Shifted down slightly

		for line in lines:
			# Get line width for centering
			line_bbox = draw.textbbox((0, 0), line, font=larger_font)
			text_x = (image.width - (line_bbox[2] - line_bbox[0])) // 2

			draw.text(
				(text_x, text_y),
				line,
				font=larger_font,
				fill=(255, 255, 255),
			)
			text_y += line_height

		# Add logo if provided (top right corner)
		if logo:
			logo_margin = 20
			logo_x = image.width - logo.width - logo_margin
			image.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)

		return image

	def _add_overlay_to_image(
		self,
		image: Image.Image,
		step_number: int,
		goal_text: str,
		regular_font: ImageFont.FreeTypeFont,
		title_font: ImageFont.FreeTypeFont,
		margin: int,
		logo: Optional[Image.Image] = None,
	) -> Image.Image:
		"""Add step number and goal overlay to an image."""
		image = image.convert('RGBA')
		txt_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
		draw = ImageDraw.Draw(txt_layer)

		# Add step number (bottom left)
		step_text = str(step_number)
		step_bbox = draw.textbbox((0, 0), step_text, font=title_font)
		step_width = step_bbox[2] - step_bbox[0]
		step_height = step_bbox[3] - step_bbox[1]

		# Position step number in bottom left
		x_step = margin + 10  # Slight additional offset from edge
		y_step = image.height - margin - step_height - 10  # Slight offset from bottom

		# Draw rounded rectangle background for step number
		padding = 20  # Increased padding
		step_bg_bbox = (
			x_step - padding,
			y_step - padding,
			x_step + step_width + padding,
			y_step + step_height + padding,
		)
		draw.rounded_rectangle(
			step_bg_bbox,
			radius=15,  # Add rounded corners
			fill=(0, 0, 0, 255),
		)

		# Draw step number
		draw.text(
			(x_step, y_step),
			step_text,
			font=title_font,
			fill=(255, 255, 255, 255),
		)

		# Draw goal text (centered, bottom)
		max_width = image.width - (4 * margin)
		wrapped_goal = self._wrap_text(goal_text, title_font, max_width)
		goal_bbox = draw.multiline_textbbox((0, 0), wrapped_goal, font=title_font)
		goal_width = goal_bbox[2] - goal_bbox[0]
		goal_height = goal_bbox[3] - goal_bbox[1]

		# Center goal text horizontally, place above step number
		x_goal = (image.width - goal_width) // 2
		y_goal = y_step - goal_height - padding * 4  # More space between step and goal

		# Draw rounded rectangle background for goal
		padding_goal = 25  # Increased padding for goal
		goal_bg_bbox = (
			x_goal - padding_goal,  # Remove extra space for logo
			y_goal - padding_goal,
			x_goal + goal_width + padding_goal,
			y_goal + goal_height + padding_goal,
		)
		draw.rounded_rectangle(
			goal_bg_bbox,
			radius=15,  # Add rounded corners
			fill=(0, 0, 0, 255),
		)

		# Draw goal text
		draw.multiline_text(
			(x_goal, y_goal),
			wrapped_goal,
			font=title_font,
			fill=(255, 255, 255, 255),
			align='center',
		)

		# Add logo if provided (top right corner)
		if logo:
			logo_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
			logo_margin = 20
			logo_x = image.width - logo.width - logo_margin
			logo_layer.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)
			txt_layer = Image.alpha_composite(logo_layer, txt_layer)

		# Composite and convert
		result = Image.alpha_composite(image, txt_layer)
		return result.convert('RGB')

	def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
		"""
		Wrap text to fit within a given width.

		Args:
			text: Text to wrap
			font: Font to use for text
			max_width: Maximum width in pixels

		Returns:
			Wrapped text with newlines
		"""
		words = text.split()
		lines = []
		current_line = []

		for word in words:
			current_line.append(word)
			line = ' '.join(current_line)
			bbox = font.getbbox(line)
			if bbox[2] > max_width:
				if len(current_line) == 1:
					lines.append(current_line.pop())
				else:
					current_line.pop()
					lines.append(' '.join(current_line))
					current_line = [word]

		if current_line:
			lines.append(' '.join(current_line))

		return '\n'.join(lines)

	def _create_frame(
		self, screenshot: str, text: str, step_number: int, width: int = 1200, height: int = 800
	) -> Image.Image:
		"""Create a frame for the GIF with improved styling"""

		# Create base image
		frame = Image.new('RGB', (width, height), 'white')

		# Load and resize screenshot
		screenshot_img = Image.open(BytesIO(base64.b64decode(screenshot)))
		screenshot_img.thumbnail((width - 40, height - 160))  # Leave space for text

		# Calculate positions
		screenshot_x = (width - screenshot_img.width) // 2
		screenshot_y = 120  # Leave space for header

		# Draw screenshot
		frame.paste(screenshot_img, (screenshot_x, screenshot_y))

		# Load browser-use logo
		logo_size = 100  # Increased size for browser-use logo
		logo_path = os.path.join(os.path.dirname(__file__), 'assets/browser-use-logo.png')
		if os.path.exists(logo_path):
			logo = Image.open(logo_path)
			logo.thumbnail((logo_size, logo_size))
			frame.paste(
				logo, (width - logo_size - 20, 20), logo if 'A' in logo.getbands() else None
			)

		# Create drawing context
		draw = ImageDraw.Draw(frame)

		# Load fonts
		try:
			title_font = ImageFont.truetype('Arial.ttf', 36)  # Increased font size
			text_font = ImageFont.truetype('Arial.ttf', 24)  # Increased font size
			number_font = ImageFont.truetype('Arial.ttf', 48)  # Increased font size for step number
		except:
			title_font = ImageFont.load_default()
			text_font = ImageFont.load_default()
			number_font = ImageFont.load_default()

		# Draw task text with increased spacing
		margin = 80  # Increased margin
		max_text_width = width - (2 * margin)

		# Create rounded rectangle for goal text
		text_padding = 20
		text_lines = textwrap.wrap(text, width=60)
		text_height = sum(draw.textsize(line, font=text_font)[1] for line in text_lines)
		text_box_height = text_height + (2 * text_padding)

		# Draw rounded rectangle background for goal
		goal_bg_coords = [
			margin - text_padding,
			40,  # Top position
			width - margin + text_padding,
			40 + text_box_height,
		]
		draw.rounded_rectangle(
			goal_bg_coords,
			radius=15,  # Increased radius for more rounded corners
			fill='#f0f0f0',
		)

		# Draw browser-use small logo in top left of goal box
		small_logo_size = 30
		if os.path.exists(logo_path):
			small_logo = Image.open(logo_path)
			small_logo.thumbnail((small_logo_size, small_logo_size))
			frame.paste(
				small_logo,
				(margin - text_padding + 10, 45),  # Positioned inside goal box
				small_logo if 'A' in small_logo.getbands() else None,
			)

		# Draw text with proper wrapping
		y = 50  # Starting y position for text
		for line in text_lines:
			draw.text((margin + small_logo_size + 20, y), line, font=text_font, fill='black')
			y += draw.textsize(line, font=text_font)[1] + 5

		# Draw step number with rounded background
		number_text = str(step_number)
		number_size = draw.textsize(number_text, font=number_font)
		number_padding = 20
		number_box_width = number_size[0] + (2 * number_padding)
		number_box_height = number_size[1] + (2 * number_padding)

		# Draw rounded rectangle for step number
		number_bg_coords = [
			20,  # Left position
			height - number_box_height - 20,  # Bottom position
			20 + number_box_width,
			height - 20,
		]
		draw.rounded_rectangle(
			number_bg_coords,
			radius=15,
			fill='#007AFF',  # Blue background
		)

		# Center number in its background
		number_x = number_bg_coords[0] + ((number_box_width - number_size[0]) // 2)
		number_y = number_bg_coords[1] + ((number_box_height - number_size[1]) // 2)
		draw.text((number_x, number_y), number_text, font=number_font, fill='white')

		return frame

	async def get_next_action(self, input_messages: list[dict]) -> AgentOutput:
		"""Get the next action from the LLM using Tangent's structured output format.
		
		This method interfaces with Tangent's LLM client to get the next action,
		ensuring the response is properly formatted as a structured output.
		
		Args:
			input_messages: List of conversation messages to send to the LLM
			
		Returns:
			An AgentOutput instance containing the LLM's response in structured format
			
		Raises:
			ValueError: If the response cannot be parsed into the structured format
		"""
		# Use tangent client to run the agent
		response: Response = await self.llm.run(
			agent=TangentAgent(
				name="BrowserAgent",
				model="gpt-4o",
				instructions=self.message_manager.system_prompt.get_prompt(),
			),
			messages=input_messages,
			conversation_id=self.conversation_id,
			stream=False,
		)
		
		try:
			# Parse the response using the AgentOutput model
			output = self.AgentOutput.model_validate_json(response.messages[0]['content'])
			return output
		except Exception as e:
			raise ValueError(f"Could not parse response: {e}")

	async def load_conversation(self, conversation_id: str) -> None:
		"""Load an existing conversation state"""
		try:
			await self.message_manager.load_conversation(conversation_id)
			self.conversation_id = conversation_id
			logger.info(f"Loaded conversation: {conversation_id}")
		except Exception as e:
			logger.error(f"Failed to load conversation {conversation_id}: {e}")
			# Keep the current conversation ID if loading fails
			pass