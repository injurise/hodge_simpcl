from .augmentor import SAugmentor, Compose, RandomChoice,GAugmentor,GCompose,GRandomChoice
from .identity import Identity
from .spec_edge_drop import SpectralEdgeDrop

__all__ = [
    'SAugmentor',
    'Compose',
    'RandomChoice',
    'Identity',
    "SpectralEdgeDrop"
]

classes = __all__

