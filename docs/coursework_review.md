# Review of the original coursework code

**Assignment:** *Preprocessing Polygonal Dataset for Paddy Field Insect Segmentation using Mask-RCNN* (BSBI Berlin, M.Sc. Data Analytics, submitted 29 January 2025).

**Original form:** a Google Colab notebook using Detectron2, shown as screenshots in the written report. This review covers that notebook and is the reason for the rebuild in this repository.

## Summary

| Area | Rating | Main issue |
|---|---|---|
| Correctness | Low | A box-only model was trained for a segmentation task; validation reused the training data. |
| Brief coverage | Partial | Augmentation and metrics were described in prose but not implemented. |
| Code structure | Low | Notebook cells with global state, hard-coded Colab paths, `os.chdir` into the library folder. |
| Robustness | Low | No validation of the annotation file; failures surface as library stack traces. |
| Reproducibility | Low | No seed, no saved split, no pinned environment; weights were not kept. |
| Readability | Medium | Comments are present and the steps follow the brief's order. |

## Findings

1. **Wrong model for the task.** The config was `COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml`. Faster R-CNN has no mask head, so the trained model could not produce segmentations. The brief asks for Mask R-CNN (`COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` in Detectron2).
2. **Data leakage.** `paddy_pests_train` and `paddy_pests_val` were both registered on `Paddy Pest.json`, so evaluation ran on the training images.
3. **Augmentation not applied.** The brief requires flips, scale, rotation and brightness/contrast applied consistently to images and masks. The notebook used Detectron2's default resize only.
4. **Metrics not computed.** IoU, mAP and pixel accuracy are explained in the report, but no numbers are reported. `trainer.test` was called, but its output was not interpreted.
5. **Class metadata mismatch.** The annotations' category is named `pankaj`, while `thing_classes` was set to `["Results"]`. The model trains, but the labels in visualisations are wrong.
6. **Training settings.** `IMS_PER_BATCH = 8` on a single T4 risks running out of memory with Mask R-CNN; there was no learning-rate schedule, no checkpoint selection and no fixed seed.
7. **Environment coupling.** `!pip install`, `!git clone`, `%cd` and `/content/...` paths tie the code to one Colab session.

## What was good

- The polygon annotation work itself (112 images, 127 instances in CVAT) is careful and is reused here unchanged apart from the category name.
- The choice of transfer learning from COCO weights is right for a dataset this small.
- The report correctly explains why polygons beat boxes for irregular insect shapes and why pixel accuracy flatters background-heavy images.

## How each finding is addressed

| Finding | Fix in this repository |
|---|---|
| 1 | `paddyseg.model.build_model` builds torchvision Mask R-CNN with box and mask predictors. |
| 2 | `paddyseg split` writes disjoint train/val/test files by image; a test checks they are disjoint. |
| 3 | `paddyseg.transforms` applies every geometric transform to image and masks together; a test checks the alignment. |
| 4 | `paddyseg.metrics.evaluate` reports IoU, pixel accuracy, P/R/F1 and COCO mAP. |
| 5 | Class names come from the annotation file; the category is renamed `insect`. |
| 6 | Batch size 2, step learning-rate decay, gradient clipping, NaN guard, best checkpoint on validation loss, seeded runs. |
| 7 | Installable package and CLI; no notebook magics or hard-coded paths. |
