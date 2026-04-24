from .config import (
    LossConfig,
    ModelConfig,
    OptimizationConfig,
    PatchConfig,
    TrainingConfig,
    ValidationConfig,
)

__all__ = [
    "LossConfig",
    "ModelConfig",
    "OptimizationConfig",
    "PatchConfig",
    "ResUNet",
    "TrainingConfig",
    "ValidationConfig",
    "build_model",
]


def __getattr__(name: str):
    if name == "ResUNet":
        from .model import ResUNet

        return ResUNet
    if name == "build_model":
        from .model import build_model

        return build_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
