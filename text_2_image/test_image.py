#https://huggingface.co/runwayml/stable-diffusion-v1-5

from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
#pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
#pipe = pipe.to("cuda")
pipe = pipe.to("cpu")
#pipe.enable_model_cpu_offload()

#prompt = "a photo of a programmer fighting a dragon"
prompt = "a picture of a dragon and a scientist in a forest"
image = pipe(prompt).images[0]  

#image.save("dragon_and_programmer.png")    
image.save("dragon_and_nerd.png")