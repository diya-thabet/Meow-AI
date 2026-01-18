from PIL import Image
import io

def read_image_file(file_data: bytes) -> Image.Image:
    """Converts uploaded bytes to a PIL Image."""
    try:
        image = Image.open(io.BytesIO(file_data))
        image = image.convert("RGB") # Ensure 3 channels
        return image
    except Exception as e:
        raise ValueError(f"Invalid image format: {e}")