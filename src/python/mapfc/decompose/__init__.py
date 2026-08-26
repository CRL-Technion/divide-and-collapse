"""Reach sets, interaction graph H, connected-component decomposition."""

from mapfc.decompose.components import connected_components
from mapfc.decompose.interaction import edge_witnesses, interaction_graph
from mapfc.decompose.pipeline import DecompositionResult, decompose
from mapfc.decompose.reach import reach_set

__all__ = [
    "DecompositionResult",
    "connected_components",
    "decompose",
    "edge_witnesses",
    "interaction_graph",
    "reach_set",
]
