import os
import replicate
from typing import Literal
from pathlib import Path
import time

def generate_image(
    prompt: str,
    aspect_ratio: Literal["1:1", "3:2", "2:3", "16:9", "9:16"] = "3:2",
) -> str:
    """Generate an image using Replicate's Flux model.
    
    Args:
        prompt (str): Description of the image to generate
        aspect_ratio (str): Aspect ratio of the output image. Options: 1:1, 3:2, 2:3, 16:9, 9:16
    
    Returns:
        str: Path to the generated image file
    """
    
    # Validate environment
    if "REPLICATE_API_TOKEN" not in os.environ:
        raise ValueError("REPLICATE_API_TOKEN environment variable is not set")
    
    # Set up img directory in project root
    img_dir = Path("img")
    img_dir.mkdir(exist_ok=True)
    
    # Create filename with timestamp
    timestamp = int(time.time())
    filename = f"flux_image_{timestamp}.jpg"
    output_path = img_dir / filename
    
    # Generate image
    input_params = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "safety_tolerance": 6,  # Maximum creative freedom
        "raw": True  # Less processed, more natural-looking images
    }
    
    try:
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro-ultra",
            input=input_params
        )
        
        # Save the image
        with open(output_path, "wb") as file:
            file.write(output.read())
            
        return str(output_path)
        
    except Exception as e:
        raise Exception(f"Image generation failed: {str(e)}")
