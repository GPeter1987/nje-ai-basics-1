"""
1. modell – ételfelismerés (ViT, Food-101).  Felelős: A

Bemenet: PIL Image (RGB).  Kimenet: Recognition (top-1 + top-3, bizonyosság).
"""

from PIL import Image

from pipeline.schemas import Recognition


def recognize(image: Image.Image) -> Recognition:
    """Felismeri az ételt a képen, és visszaadja a top-3 jelöltet."""
    raise NotImplementedError("TODO (A): ViT betöltése és osztályozás")
