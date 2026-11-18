import cv2
import numpy as np
from PIL import Image
from carvekit.api.high import HiInterface
import torch
import os


class MaskGenerator:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = HiInterface(
            object_type="object",
            batch_size_seg=1,
            batch_size_matting=1,
            device=self.device,
            seg_mask_size=640,
            matting_mask_size=2048,
            trimap_prob_threshold=10,
            trimap_dilation=10,  # Reduced for tighter masks
            trimap_erosion_iters=10,
            fp16=False
        )

    def generate_mask(self, input_path, mask_path, show=False):
        """Generate foreground mask using Big-LAMA"""
        try:
            # 1. Load image
            img = cv2.imread(input_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {input_path}")

            # 2. Generate mask automatically
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            print("Generating foreground mask...")
            result = self.model([img_pil])

            if not result:
                raise RuntimeError("Foreground detection failed")

            # 3. Extract mask from result
            mask = np.array(result[0])[:, :, 3]  # Alpha channel
            mask = (mask > 127).astype(np.uint8) * 255  # Binarize

            # 4. Save mask
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            cv2.imwrite(mask_path, mask)
            print(f"Mask generated and saved to {mask_path}")

            # 5. Show results (only when requested)
            if show:
                self._display_results(img, mask)

            return mask

        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def _display_results(self, original, mask):
        """Show original image and generated mask"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Generated Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    generator = MaskGenerator()
    generator.generate_mask(
        input_path="images/input.jpg",  # image path
        mask_path="images/mask.png",  # Output mask path
        show=True
    )




