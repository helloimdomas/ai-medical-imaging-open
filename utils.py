"""Shared helpers used across multiple pipeline scripts."""

import base64
import io
from datetime import datetime
from pathlib import Path


def backup_if_exists(path: Path):
    """Rename existing file with a timestamp suffix to avoid overwriting."""
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        path.rename(backup)
        print(f"Backed up existing file to: {backup}")


def image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
