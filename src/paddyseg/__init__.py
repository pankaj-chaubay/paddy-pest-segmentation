"""paddyseg: polygon-annotated insect segmentation in paddy field images with Mask R-CNN.

The package covers the full path from a CVAT COCO 1.0 export to an evaluated model:

* :mod:`paddyseg.coco` loads, validates, summarises and splits COCO annotation files.
* :mod:`paddyseg.transforms` builds augmentation pipelines that move images and masks together.
* :mod:`paddyseg.dataset` turns annotations into the tensors Mask R-CNN expects.
* :mod:`paddyseg.model` builds a torchvision Mask R-CNN with a new prediction head.
* :mod:`paddyseg.engine` trains, validates and predicts.
* :mod:`paddyseg.metrics` computes IoU, pixel accuracy, precision/recall and COCO mAP.
"""

from paddyseg.errors import AnnotationError, DatasetError, PaddySegError

__all__ = ["AnnotationError", "DatasetError", "PaddySegError", "__version__"]
__version__ = "1.0.0"
