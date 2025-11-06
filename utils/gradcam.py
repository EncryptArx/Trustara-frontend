from PIL import Image, ImageOps
import numpy as np


def generate_gradcam_overlay(input_path: str, output_path: str) -> None:
    """
    Placeholder Grad-CAM overlay generator.
    For MVP scaffolding, it tints the image with a red heat overlay.
    """
    img = Image.open(input_path).convert("RGB")
    heat = Image.new("RGB", img.size, (255, 0, 0))
    heat = ImageOps.colorize(ImageOps.grayscale(heat), black=(0, 0, 0), white=(255, 0, 0))
    heat_arr = np.asarray(heat).astype(np.float32)
    img_arr = np.asarray(img).astype(np.float32)
    overlay = (0.6 * img_arr + 0.4 * heat_arr).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)


