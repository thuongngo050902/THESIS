"""Network package for MAT generator and discriminator modules."""

from .structure_guidance import (
    MaskSeverityGate,
    StructureAwareAttentionBias,
    StructureEncoder,
    StructureInputBuilder,
    StructureResidualAdapter,
)

__all__ = [
    "MaskSeverityGate",
    "StructureAwareAttentionBias",
    "StructureEncoder",
    "StructureInputBuilder",
    "StructureResidualAdapter",
]