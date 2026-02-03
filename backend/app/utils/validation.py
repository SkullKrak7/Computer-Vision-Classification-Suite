"""Image upload validation utilities"""

import io
from typing import Tuple

from fastapi import HTTPException, UploadFile, status
from PIL import Image

# Security constants per Gold Standard
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_DIMENSION = 4096  # Prevent decompression bombs
MIN_IMAGE_DIMENSION = 32


async def validate_image_upload(file: UploadFile) -> Tuple[Image.Image, str]:
    """
    Validate uploaded image file per Gold Standard security requirements.

    Args:
        file: Uploaded file from FastAPI

    Returns:
        Tuple of (PIL Image, filename)

    Raises:
        HTTPException: If validation fails
    """
    # Validate filename exists
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required")

    # Validate file extension
    file_ext = file.filename.lower().split(".")[-1] if "." in file.filename else ""
    if f".{file_ext}" not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type. Allowed: {', '.join(ALLOWED_MIME_TYPES)}",
        )

    # Read file content
    contents = await file.read()

    # Validate file size
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB",
        )

    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    # Validate image format and prevent decompression bombs
    try:
        image = Image.open(io.BytesIO(contents))

        # Verify image format matches extension
        if image.format.lower() not in {"jpeg", "jpg", "png"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image format: {image.format}")

        # Validate image dimensions (prevent decompression bombs)
        width, height = image.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image dimensions too large. Maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}",
            )

        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image dimensions too small. Minimum: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}",
            )

        # Convert to RGB if needed (handle RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image, file.filename

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image file: {str(e)}") from e


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = filename.split("/")[-1].split("\\")[-1]

    # Remove dangerous characters
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    sanitized = "".join(c if c in allowed_chars else "_" for c in filename)

    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]

    return sanitized or "unnamed_file"
