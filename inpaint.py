from simple_lama_inpainting import SimpleLama
from PIL import Image
import cv2
import numpy as np
import torch

__all__ = ["SimpleLama"]


def display_results(original_img, inpainted_img):
    """Show original image and inpainted result"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    original_display = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    inpainted_display = cv2.cvtColor(np.array(inpainted_img), cv2.COLOR_RGB2BGR)

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_display, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Inpainted Result")
    plt.imshow(cv2.cvtColor(inpainted_display, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # === Runtime Setup === #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cpu = (device.type == "cpu")

    # === Paths === #
    img_path = "images/input.jpg"
    mask_path = "images/mask.png"

    # === Load Image and Mask === #
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    w, h = img.size

    # === Conditional Resize (CPU + High Resolution) === #
    if use_cpu and (w >= 1024 and h >= 1024):
        print("⚠ GPU not available. Running on CPU.")
        print("📐 Input resolution is high (≥1024x1024). Resizing for performance.")

        fallback_size = (640, 480)  # or (512, 512) depending on memory
        img = img.resize(fallback_size, Image.LANCZOS)
        mask = mask.resize(fallback_size, Image.NEAREST)
    else:
        print(f"✅ No resizing needed. Using original resolution: {w}x{h}")

    # === Prepare Mask === #
    mask_np = np.array(mask)
    mask_np = np.where(mask_np > 128, 255, 0).astype(np.uint8)
    kernel = np.ones((15, 15), np.uint8)
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    mask = Image.fromarray(mask_np)

    # === Ensure Mask and Image Match === #
    if mask.size != img.size:
        print(f"Resizing mask from {mask.size} to {img.size}")
        mask = mask.resize(img.size, Image.NEAREST)

    # === Inpainting === #
    simple_lama = SimpleLama()
    res = simple_lama(img, mask)

    # === Save and Display === #
    res = res.crop((0, 0, img.size[0], img.size[1]))
    res.save("images/inpainted.png")
    display_results(img, res)


if __name__ == "__main__":
    main()
