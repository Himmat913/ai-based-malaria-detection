"""Display-only image helpers.

Nothing in this module feeds the model. Inference and Grad-CAM always receive
the decoded PIL image unchanged; these helpers only prepare *copies* for the
browser so that previews keep their true proportions.
"""

import base64
import hashlib
import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

DISPLAY_MAX_SIDE = 1024   # longest side of the preview sent to the browser
HEATMAP_MIN_SIDE = 448    # Grad-CAM previews are mapped up to at least this size


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


@dataclass(frozen=True)
class DecodedUpload:
    image: Image.Image   # RGB image passed to the model, exactly as before
    digest: str          # content hash, used to tie results to this exact file
    name: str
    size_bytes: int
    width: int
    height: int
    source_format: str
    source_mode: str


def decode_upload(name: str, data: bytes) -> DecodedUpload:
    try:
        with Image.open(io.BytesIO(data)) as raw:
            raw.load()
            source_format = raw.format or "Unknown"
            source_mode = raw.mode
            image = raw.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(
            "This file could not be read as an image. It may be corrupted or not a real JPG/PNG."
        ) from exc

    return DecodedUpload(
        image=image,
        digest=hashlib.sha256(data).hexdigest(),
        name=name,
        size_bytes=len(data),
        width=image.width,
        height=image.height,
        source_format=source_format,
        source_mode=source_mode,
    )


def _encode(image: Image.Image) -> str:
    buffer = io.BytesIO()
    if max(image.size) <= 640:
        image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    else:
        image.save(buffer, format="JPEG", quality=90)
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def preview_data_uri(image: Image.Image) -> str:
    """Aspect-preserving preview, downscaled only if very large."""
    preview = image.copy()
    preview.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE), Image.LANCZOS)
    return _encode(preview)


def heatmap_display_size(original_size):
    """Size that restores the original proportions for the Grad-CAM overlay.

    The model (and therefore Grad-CAM) works on a 224 x 224 square. Mapping the
    overlay back to the source aspect ratio lets it line up with the original
    image when both are shown side by side.
    """
    width, height = original_size
    longest = max(width, height)
    target_longest = min(max(longest, HEATMAP_MIN_SIDE), DISPLAY_MAX_SIDE)
    scale = target_longest / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def map_heatmap_to_original(overlay: Image.Image, original_size) -> Image.Image:
    return overlay.resize(heatmap_display_size(original_size), Image.BICUBIC)


def to_data_uri(image: Image.Image) -> str:
    return _encode(image)


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def format_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
