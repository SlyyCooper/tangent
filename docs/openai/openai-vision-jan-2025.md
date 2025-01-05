gpt-4o Vision
======

The 'gpt-4o' model has vision capabilities, meaning the model can take in images and answer questions about them.

Quickstart
----------

Images are made available to the model in two main ways: by passing a link to the image (image on the web) or by passing the base64 encoded image directly in the request (local image upload). Images can be passed in the `user` messages.

Analyze the content of an image

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])
```

**LOCAL IMAGE UPLOADING (use base64 encoded images)**
-----------------------------------

If you have an image or set of images locally, you can pass those to the model in base 64 encoded format, here is an example of this in action:

```python
import base64
from openai import OpenAI

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Path to your image
image_path = "path_to_your_image.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)

print(response.choices[0])
```

Multiple image inputs
---------------------

The Chat Completions API is capable of taking in and processing multiple image inputs in both base64 encoded format or as an image URL. The model will process each image and use the information from all of them to answer the question.

Multiple image inputs

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are in these images? Is there any difference between them?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0])
```
Here the model is shown two copies of the same image and can answer questions about both or each of the images independently.

**High fidelity image understanding**
----------------------------------------

*   `high` will enable "high res" mode, which first allows the model to first see the low res image (using 85 tokens) and then creates detailed crops using 170 tokens for each 512px x 512px tile.

Use **ONLY** the high detail level

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        "detail": "high",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)
```

**Video understanding with vision**
-------------------------------------

While the GPT-4o doesn't take videos as input directly, we can use vision and the 128K context window to describe the static frames of a whole video at once.

```python
from tangent import tangent, Agent, run_tangent_loop
import cv2
import base64
import time
from typing import List, Dict

class VideoAgent(Agent):
    def __init__(self, video_path: str):
        super().__init__(
            name="Video Analysis Agent",
            model="gpt-4o",
            instructions="You analyze video frames and engage in natural conversation about the video content."
        )
        self.frames = self._process_video(video_path)
        self.frame_context = []

    def _process_video(self, video_path: str) -> List[str]:
        video = cv2.VideoCapture(video_path)
        frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))
        video.release()
        return frames

    def get_context(self, query: str) -> List[Dict]:
        # Sample every 50th frame for context
        sampled_frames = self.frames[::50]
        return [
            {"type": "text", "text": query},
            *[{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}",
                    "detail": "high"
                }
            } for frame in sampled_frames]
        ]

# Usage
client = tangent()
agent = VideoAgent("data/bison.mp4")
run_tangent_loop(agent, stream=True, debug=False)
```