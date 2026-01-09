# -*- coding: utf-8 -*-
from .prediction import router as prediction_router
from .scl90 import router as scl90_router

__all__ = ["prediction_router", "scl90_router"]
