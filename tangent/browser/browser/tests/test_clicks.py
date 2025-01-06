import asyncio
import json
import os
from pathlib import Path

import pytest
import pytest_asyncio

from tangent.browser.browser.browser import Browser, BrowserConfig
from tangent.browser.dom.views import ElementTreeSerializer
from tangent.browser.utils import time_execution_sync


@pytest_asyncio.fixture
async def browser():
	browser_service = Browser(config=BrowserConfig(headless=False, disable_security=True))
	async with await browser_service.new_context() as context:
		yield context
	await browser_service.close()


@pytest.mark.asyncio
async def test_highlight_elements(browser):
	page = await browser.get_current_page()
	await page.goto('https://huggingface.co/')

	await asyncio.sleep(1)

	try:
		state = await browser.get_state()

		# Create output directory if it doesn't exist
		output_dir = Path(__file__).parent.parent.parent.parent / 'tmp'
		output_dir.mkdir(exist_ok=True)
		output_file = output_dir / 'page.json'

		with open(output_file, 'w') as f:
			json.dump(
				ElementTreeSerializer.dom_element_node_to_json(state.element_tree),
				f,
					indent=1,
			)

		# Find and print duplicate XPaths
		xpath_counts = {}
		if not state.selector_map:
			pytest.skip("No elements found in selector map")
			
		for selector in state.selector_map.values():
			xpath = selector.xpath
			if xpath in xpath_counts:
				xpath_counts[xpath] += 1
			else:
				xpath_counts[xpath] = 1

		print('\nDuplicate XPaths found:')
		for xpath, count in xpath_counts.items():
			if count > 1:
				print(f'XPath: {xpath}')
				print(f'Count: {count}\n')

		print(list(state.selector_map.keys()), 'Selector map keys')
		print(state.element_tree.clickable_elements_to_string())
		
		# Test clicking the first element
		if state.selector_map:
			first_key = list(state.selector_map.keys())[0]
			node_element = state.selector_map[first_key]
			await browser._click_element_node(node_element)

	except Exception as e:
		pytest.fail(f"Test failed: {str(e)}")
