from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
controlnet = ControlNetModel.from_pretrained('./new_models')

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet
)

pipe = pipe.to("cuda")

from PIL import Image
pose_image = Image.open("./conditioning_image_1.png").convert("RGB")

image = pipe("red circle with blue background", pose_image, num_inference_steps=20, generator=None)

