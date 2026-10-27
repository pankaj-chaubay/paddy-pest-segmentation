"""Mask R-CNN model construction."""

from __future__ import annotations

from torchvision.models.detection import MaskRCNN, MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(
    num_classes: int,
    *,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 800,
    max_size: int = 1333,
    mask_hidden_channels: int = 256,
) -> MaskRCNN:
    """Build a ResNet-50-FPN Mask R-CNN with box and mask heads sized for ``num_classes``.

    With ``pretrained=True`` the whole network starts from COCO weights (transfer learning)
    and only the two class-specific predictors are replaced. The backbone's earliest stages
    stay frozen according to ``trainable_backbone_layers``.

    Args:
        num_classes: Number of classes **including** the background.
        pretrained: Start from COCO-trained weights. They are downloaded on first use.
        trainable_backbone_layers: How many ResNet stages (0-5) are fine-tuned.
        min_size: Shorter image side after the model's internal resize.
        max_size: Upper bound for the longer side after resizing.
        mask_hidden_channels: Width of the mask predictor's hidden layer.

    Returns:
        A ``torchvision`` :class:`~torchvision.models.detection.MaskRCNN` in training mode.

    Raises:
        ValueError: If ``num_classes`` is below 2 or ``trainable_backbone_layers`` is out of range.
    """
    if num_classes < 2:
        raise ValueError("num_classes counts the background, so it must be at least 2")
    if not 0 <= trainable_backbone_layers <= 5:
        raise ValueError("trainable_backbone_layers must be between 0 and 5")

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        # Freezing only makes sense for pretrained weights; random weights train everything.
        trainable_backbone_layers=trainable_backbone_layers if pretrained else None,
        min_size=min_size,
        max_size=max_size,
    )

    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, num_classes)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, mask_hidden_channels, num_classes)
    return model
