"""
A lánc összefűzése: felismerés -> elemzés -> tápérték -> ajánlás.  Felelős: A

Lépésenkénti időmérés, hibakezelés és pipeline-napló (F9).
"""

from PIL import Image

from pipeline.schemas import PipelineResult, RecommendMode


def run_pipeline(image: Image.Image, grams: float, mode: RecommendMode = "similar") -> PipelineResult:
    """Lefuttatja a teljes láncot egy képre."""
    raise NotImplementedError("TODO (A): lánc, időmérés, hibakezelés")
