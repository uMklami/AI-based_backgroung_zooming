import cv2
import numpy as np

MAX_ZOOM = 4.0


class ImageZoom:
    def __init__(self, upscaler=None):
        """upscaler: optional SmartUpscaler; falls back to Lanczos when None."""
        self.upscaler = upscaler
        print("Image zoom processor initialized")

    def process_image(self, image_path, zoom_factor=1.70, output_path=None,
                      show=False, zoom_rect=None, fixed_point=None):
        """
        Zoom the input image.
        - zoom_rect: optional normalized dict {x, y, w, h} (0-1) selecting the
          background region to zoom toward. The region (aspect-corrected)
          fills the frame; zoom_factor is then ignored.
        - fixed_point: optional (x, y) pixel that must map to itself under
          the zoom (e.g. the subject's ground-contact point, so contact
          cues stay aligned). Internally converted to the crop center
          c = p·(1 − 1/z) + s/2, then clamped.
        - With neither: classic center zoom by zoom_factor.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image from {image_path}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if zoom_rect:
                zoom_factor, center = self.rect_to_zoom(img_rgb.shape, zoom_rect)
                print(f"Region zoom: factor={zoom_factor:.2f}, center={center}")
                zoomed_img = self.zoom_image(img_rgb, zoom_factor, center=center)
            elif fixed_point is not None:
                center = self.fixed_point_to_center(img_rgb.shape, zoom_factor,
                                                    fixed_point)
                print(f"Subject-anchored zoom: fixed point {fixed_point} "
                      f"-> crop center {center}")
                zoomed_img = self.zoom_image(img_rgb, zoom_factor, center=center)
            else:
                zoomed_img = self.zoom_image(img_rgb, zoom_factor)

            if output_path:
                zoomed_bgr = cv2.cvtColor(zoomed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, zoomed_bgr)
                print(f"Zoomed image saved to {output_path}")

            if show:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.title("Original Image")
                plt.imshow(img_rgb)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.title(f"Zoomed {zoom_factor:.2f}x Image")
                plt.imshow(zoomed_img)
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            return zoomed_img

        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    @staticmethod
    def fixed_point_to_center(shape, zoom, point):
        """Crop center such that `point` maps to itself under the zoom."""
        h, w = shape[:2]
        crop_w, crop_h = w / zoom, h / zoom
        cx = point[0] * (1.0 - 1.0 / zoom) + crop_w / 2.0
        cy = point[1] * (1.0 - 1.0 / zoom) + crop_h / 2.0
        return (cx, cy)

    @staticmethod
    def rect_to_zoom(shape, rect):
        """
        Convert a normalized selection rect into (zoom_factor, center_px).
        The crop window is the smallest image-aspect window containing the rect.
        """
        h, w = shape[:2]
        rx = float(rect['x']) * w
        ry = float(rect['y']) * h
        rw = max(float(rect['w']) * w, 1.0)
        rh = max(float(rect['h']) * h, 1.0)
        cx = rx + rw / 2.0
        cy = ry + rh / 2.0

        aspect = w / h
        crop_w = max(rw, rh * aspect)
        crop_w = min(crop_w, w)
        zoom = max(1.0, min(w / crop_w, MAX_ZOOM))
        return zoom, (cx, cy)

    def zoom_image(self, img_rgb, zoom_factor, center=None):
        """High-quality zoom cropping around `center` (defaults to image center)."""
        h, w = img_rgb.shape[:2]

        crop_h = max(int(h / zoom_factor), 1)
        crop_w = max(int(w / zoom_factor), 1)

        if center is None:
            cx, cy = w / 2.0, h / 2.0
        else:
            cx, cy = center

        # Clamp the crop window inside the image
        start_x = int(round(cx - crop_w / 2.0))
        start_y = int(round(cy - crop_h / 2.0))
        start_x = max(0, min(start_x, w - crop_w))
        start_y = max(0, min(start_y, h - crop_h))

        cropped = img_rgb[start_y:start_y + crop_h, start_x:start_x + crop_w]

        # Upscale back to original size (AI upscaler if available)
        if self.upscaler is not None:
            return self.upscaler.upscale(cropped, w, h)
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)


def main():
    input_image = "images/inpainted.png"
    output_image = "images/zoomed_image.jpg"

    processor = ImageZoom()
    result = processor.process_image(input_image, zoom_factor=1.70,
                                     output_path=output_image, show=True)

    if result is not None:
        print("Processing completed successfully!")


if __name__ == "__main__":
    main()
