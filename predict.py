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
        # self.model = torch.load("./weights.pth")
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

    def resize_image(self, image, max_size=1280):
        """Görseli orantıyı koruyarak max_size altına küçültür."""
        w, h = image.size
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)  # En büyük boyutu 2048'e çekiyoruz
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        return image

    def predict(
        self,
        input_image: Path = Input(description="Interior image input"),
        prompt: str = Input(description="Prompt input"),
        sampling_steps: int = Input(
            description="How many steps to be generated", default=20, le=100, ge=1
        ),
        safety_checker: bool = Input(
            description="Disable safety checker?", default=False
        ),
    ) -> Path:

        # Safety Checker Ayarı (None = Kapalı, Default OpenAI'nin kendi fonksiyonu)
        self.pipe.safety_checker = None if safety_checker else self.pipe.safety_checker

        image = Image.open(input_image).convert("RGB")
        image = self.resize_image(image)

        image = load_image(image)  # ✅ Eksik olan load_image eklendi
        image = self.mlsd_detector(image)

        output = self.pipe(prompt, image=image, num_inference_steps=sampling_steps)

        output_image = output.images[0].convert("RGB")
        output_path = "output.png"
        output_image.save(output_path)

        return Path(output_path)
