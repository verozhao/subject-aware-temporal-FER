import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading pre-trained model: {model_id} ...")

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)
pipe = pipe.to(device)

if hasattr(pipe, 'safety_checker'):
    pipe.safety_checker = None

pipe.enable_attention_slicing() 

def augment_with_sd(image_path, output_dir, num_samples=5):
    img_path = Path(image_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    init_image = Image.open(image_path).convert("RGB")
    init_image = init_image.resize((512, 512)) 

    print(f"Processing: {img_path.name}")
    prompt = "a high quality photo of a face, realistic, 8k, detailed texture"
    negative_prompt = "cartoon, drawing, anime, ugly, deformed, blurry, bad anatomy, nsfw, nude"

    strengths = np.linspace(0.2, 0.5, num_samples).tolist()

    for i, s in enumerate(strengths):

        with torch.autocast("cuda"):
            images = pipe(
                prompt=prompt, 
                image=init_image, 
                strength=s, 
                guidance_scale=7.5,
                negative_prompt=negative_prompt,
                num_images_per_prompt=1
            ).images

        save_name = out_path / f"{img_path.stem}_sd_str{s:.2f}.jpg"
        images[0].save(save_name)
        print(f"  -> Saved: {save_name} (Strength: {s:.2f})")

if __name__ == "__main__":
    TEST_IMAGE = "C://794project_dataset//RAF_DB//test//angry//aug_226488.png"
    augment_with_sd(TEST_IMAGE, "../samples/faceid_gan_samples", num_samples=5)