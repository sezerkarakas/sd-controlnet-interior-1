# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
)
from PIL import Image
from controlnet_aux import MLSDdetector
from diffusers.utils import load_image


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mlsd_detector = MLSDdetector.from_pretrained("lllyasviel/ControlNet")
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-mlsd", torch_dtype=torch.float32
        ).to(self.device)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.safety_checker = None

    def resize_image(self, image, max_size=2048):
        """Görseli orantıyı koruyarak max_size altına küçültür."""
        w, h = image.size
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        return image

    def predict(
        self,
        input_image: Path = Input(description="Interior image input"),
        prompt: str = Input(description="Prompt input"),
        negative_prompt: str = Input(description="Negative prompt input", default=""),
        sampling_steps: int = Input(
            description="How many steps to be generated", default=20, le=100, ge=1
        ),
        safety_checker: bool = Input(
            description="Disable safety checker?", default=False
        ),
        auto_resize: bool = Input(
            description="Automatically resizes image to optimize performance and quality",
            default=True,
        ),
        width: int = Input(
            description="Width of the image if you don't want to auto resize",
            default=None,
        ),
        height: int = Input(
            description="Heigth of the image if you don't want to auto resize",
            default=None,
        ),
    ) -> Path:

        self.pipe.safety_checker = None if safety_checker else self.pipe.safety_checker

        image = Image.open(input_image).convert("RGB")
        if auto_resize:
            image = self.resize_image(image)
        elif width and height:
            image = image.resize((width, height), Image.LANCZOS)
        image = load_image(image)
        image = self.mlsd_detector(image)

        output = self.pipe(
            prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=sampling_steps,
        )

        output_image = output.images[0].convert("RGB")
        output_path = "output.png"
        output_image.save(output_path)

        return Path(output_path)
