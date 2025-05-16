# Common Setup - Install All Necessary Libraries
!pip install torch torchvision torchaudio matplotlib pillow numpy
!pip install diffusers transformers accelerate

# Building Diffusion Model (Stable Diffusion) Code

from typing import List
import torch # Make sure torch is imported for type hints if not globally
from PIL import Image # For type hinting List[Image.Image]
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline # DDPMScheduler not directly used here, pipeline handles it.

# --- Configuration ---
device_sd = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If on CPU, this will be very slow and might OOM.
# Consider a smaller model or fewer inference steps if on CPU, though results will degrade.

# --- Model Loading ---
def load_sd_model(model_id: str) -> StableDiffusionPipeline:
    """Load Stable Diffusion model with provided model_id."""
    print(f"Loading Stable Diffusion model: {model_id}...")
    # For CPU, you might need to remove torch_dtype=torch.float16 and revision="fp16"
    # or use a model specifically designed for CPU.
    if device_sd.type == 'cuda':
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16, # Use float16 for GPU memory saving and speed
            revision="fp16" # Use fp16 revision if available
            # use_auth_token=True # Set to True if model requires authentication (and you are logged in)
        ).to(device_sd)
    else: # Basic loading for CPU, might be slow / OOM
        print("Warning: Running Stable Diffusion on CPU. This will be very slow and memory-intensive.")
        pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device_sd)
    print("Model loaded.")
    return pipeline

# --- Image Generation ---
def generate_sd_images(
    pipe: StableDiffusionPipeline,
    prompts: List[str],
    num_inference_steps: int = 50, # Number of denoising steps
    guidance_scale: float = 7.5    # How much to adhere to the prompt
    ) -> List[Image.Image]:
    """Generate images based on provided prompts using Stable Diffusion."""
    print(f"Generating images for {len(prompts)} prompt(s)...")
    # On CUDA, autocast can provide speedups. Not typically used/needed for CPU.
    if device_sd.type == 'cuda':
      with torch.autocast("cuda"):
          images = pipe(prompts, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
    else:
      images = pipe(prompts, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
    print("Image generation complete.")
    return images

# --- Image Rendering ---
def render_sd_images(images: List[Image.Image], prompts: List[str]):
    """Plot the generated images with their prompts."""
    if not images:
        print("No images to render.")
        return
    plt.figure(figsize=(12, 6 * len(images))) # Adjust size based on number of images
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        plt.subplot(len(images), 1, i + 1) # Display images vertically
        plt.imshow(img)
        plt.title(f"Prompt: {prompt[:80]}..." if len(prompt) > 80 else f"Prompt: {prompt}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# --- Execution ---
model_id = "CompVis/stable-diffusion-v1-4" # A popular choice
# Other options: "runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1-base" (512px),
# "stabilityai/stable-diffusion-2-1" (768px, might need more VRAM)

prompts_list = [
    "A photorealistic portrait of an astronaut riding a horse on the moon, detailed, 4k",
    "Impressionist painting of a sunflower field at sunset, Van Gogh style"
]
# Keep num_prompts low if on CPU or limited VRAM.

if device_sd.type == 'cpu' and len(prompts_list) > 1:
    print("Reducing prompts to 1 for CPU execution to save resources.")
    prompts_list = [prompts_list[0]]


try:
    sd_pipe = load_sd_model(model_id)
    # Adjust num_inference_steps (e.g., 20-30 for faster, lower quality; 50 for standard)
    # Adjust guidance_scale (e.g., 7-10 is common. Higher values adhere more strictly to prompt)
    inference_steps = 30 if device_sd.type == 'cpu' else 50 # Fewer steps for CPU demo
    generated_images = generate_sd_images(sd_pipe, prompts_list, num_inference_steps=inference_steps)
    render_sd_images(generated_images, prompts_list)
except Exception as e:
    print(f"An error occurred during Stable Diffusion execution: {e}")
    print("If you are on a limited environment (like a free Colab GPU), you might be "
          "running out of memory. Try a smaller model or fewer prompts.")

# This Stable Diffusion example uses a pre-trained pipeline from Hugging Face
# to generate images from text prompts. It showcases a powerful modern approach.
# Note: First time running will download the model (several GBs).
