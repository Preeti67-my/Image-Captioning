from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load model only once
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def pipeline(image):
    image = Image.open(io.BytesIO(image)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    out = model.generate(**inputs, max_new_tokens=30)

    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption.capitalize()