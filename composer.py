import cv2
import numpy as np
import os


class ImageCompositor:
    def __init__(self):
        print("Initializing compositor...")

    def compose_images(self, original_path, foreground_path, zoomed_bg_path, output_path, show=False):
        """
        Perfectly composites foreground with zoomed background
        """
        try:
            # Load images with proper validation
            print("Loading images...")
            original = cv2.imread(original_path)
            foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)  # With alpha channel
            zoomed_bg = cv2.imread(zoomed_bg_path)

            # Proper array existence checking
            if original is None or foreground is None or zoomed_bg is None:
                missing = []
                if original is None: missing.append(original_path)
                if foreground is None: missing.append(foreground_path)
                if zoomed_bg is None: missing.append(zoomed_bg_path)
                raise ValueError(f"Failed to load image(s): {', '.join(missing)}")

            # Convert foreground to BGRA if it's BGR
            if foreground.shape[2] == 3:
                foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)

            # Ensure all images same size
            if foreground.shape[:2] != zoomed_bg.shape[:2]:
                print(
                    f"Resizing zoomed background from {zoomed_bg.shape[:2]} to match foreground {foreground.shape[:2]}")
                zoomed_bg = cv2.resize(zoomed_bg, (foreground.shape[1], foreground.shape[0]))

            # Extract foreground RGB and alpha mask
            fg_rgb = foreground[:, :, :3]

            # Defringe: pull the matte in slightly and re-feather so bright
            # background pixels caught at hair/edges don't halo.
            # Scaled with image size (fixed sizes over-treat small images
            # and under-treat large ones).
            from pipeline_core import defringe   # Eq. 12, shared with app/eval
            alpha_u8 = defringe(foreground[:, :, 3])
            alpha = alpha_u8 / 255.0  # Convert to 0-1 range

            # Create 3-channel alpha mask
            alpha_3ch = cv2.merge([alpha, alpha, alpha])

            # Composite images
            print("Blending images...")
            foreground_part = fg_rgb * alpha_3ch
            background_part = zoomed_bg * (1 - alpha_3ch)
            composite = foreground_part + background_part

            # Convert to uint8
            composite = np.clip(composite, 0, 255).astype(np.uint8)

            # Save result
            cv2.imwrite(output_path, composite)
            print(f"Composite image saved to {output_path}")

            # Display results (only when requested)
            if show:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.title("Original Image")
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title("Final Composition")
                plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
                plt.axis('off')

                plt.tight_layout()
                plt.show()

            return composite  # Return the composite image array

        except Exception as e:
            print(f"Compositing error: {str(e)}")
            return None


def main():
    # File paths - replace with your actual files
    original_img = "images/input.jpg"
    foreground_img = "images/foreground_output.png"  # Must be PNG with transparency
    zoomed_bg_img = "images/zoomed_image.jpg"
    output_img = "results/final_composite.jpg"

    # Create directories if they don't exist
    os.makedirs("images", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Verify files exist before processing
    for path in [original_img, foreground_img, zoomed_bg_img]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    # Process images
    compositor = ImageCompositor()
    result = compositor.compose_images(
        original_img,
        foreground_img,
        zoomed_bg_img,
        output_img,
        show=True
    )

    if result is not None:
        print("Compositing completed successfully!")


if __name__ == "__main__":
    main()
