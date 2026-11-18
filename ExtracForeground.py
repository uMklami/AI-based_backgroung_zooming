import cv2
import torch
import numpy as np
from PIL import Image
from carvekit.api.high import HiInterface
import os


class ForegroundExtractor:
    def __init__(self, device=None):
        # CarveKit supports cuda or cpu; EXTRACT_DEVICE overrides
        self.device = device or os.environ.get(
            'EXTRACT_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.load_model()

    def load_model(self):
        """Load CarveKit model for foreground extraction"""
        try:
            print("Initializing CarveKit model...")
            # "object" selects Tracer-B7 as the salient-object segmenter
            # (Section 3.3 of the paper); "hairs-like" would select U^2-Net,
            # which was 0.17 F1 worse on DUTS-TE (Table 7). FBA Matting
            # resolves the automatically generated trimap.
            self.interface = HiInterface(
                object_type="object",
                batch_size_seg=1,
                batch_size_matting=1,
                device=self.device,  # was missing: CarveKit defaulted to CPU!
                seg_mask_size=640,
                matting_mask_size=2048,
                trimap_prob_threshold=10,
                trimap_dilation=10,
                trimap_erosion_iters=10,
                fp16=False
            )
            print("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def extract_foreground(self, image_path):
        """Extract foreground with transparency and apply edge-preserving smoothing"""
        return self.extract_array(Image.open(image_path))

    def extract_array(self, image):
        """Same as extract_foreground, for a PIL image or RGB numpy array.
        Returns an RGBA uint8 array (F, alpha_auto) of the input size."""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = image.convert("RGBA")

            # Process through CarveKit
            print("Extracting foreground...")
            result = self.interface([image])

            if not result:
                raise RuntimeError("No foreground detected in the image.")

            # Convert to NumPy array (already RGBA)
            foreground = np.array(result[0])

            # Split into RGB and Alpha channels
            rgb = foreground[:, :, :3]
            alpha = foreground[:, :, 3]

            # --- Refine Alpha Channel ---
            # Dilate slightly to expand matte
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated_alpha = cv2.dilate(alpha, kernel, iterations=1)

            # Smooth alpha matte with small Gaussian blur
            blurred_alpha = cv2.GaussianBlur(dilated_alpha, (3, 3), 0)

            # --- Refine RGB Channels ---
            # Apply edge-preserving filter to reduce noise while keeping edges sharp
            filtered_rgb = cv2.edgePreservingFilter(
                rgb,
                flags=cv2.RECURS_FILTER,   # Best for natural images
                sigma_s=3,                # Spatial smoothness
                sigma_r=0.2                # Color smoothness
            )

            # --- Combine back into RGBA ---
            refined_foreground = np.dstack((filtered_rgb, blurred_alpha))

            return refined_foreground

        except Exception as e:
            raise RuntimeError(f"Foreground extraction error: {str(e)}")

    def save_foreground(self, foreground, output_path):
        """Save refined foreground as PNG with transparency"""
        try:
            # Convert NumPy array to PIL Image
            foreground_pil = Image.fromarray(foreground, mode='RGBA')
            foreground_pil.save(output_path, "PNG")
            print(f"Foreground saved to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Error saving foreground: {str(e)}")


def main():
    # Set input and output paths
    input_path = "images/input.jpg"      # Make sure this file exists
    output_path = "images/foreground_output.png"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize extractor
    extractor = ForegroundExtractor()

    # Process image
    try:
        # Extract and refine foreground
        refined_foreground = extractor.extract_foreground(input_path)

        # Save the final result
        extractor.save_foreground(refined_foreground, output_path)

        # Display original vs result
        import matplotlib.pyplot as plt
        original = Image.open(input_path)
        result_img = Image.open(output_path)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(original)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Refined Foreground")
        plt.imshow(result_img)
        plt.axis("off")

        plt.tight_layout()
        # plt.show()

    except Exception as e:
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    main()