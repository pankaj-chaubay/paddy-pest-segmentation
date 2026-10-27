# Data

| Path | Content |
|---|---|
| `annotations/paddy_pest_coco.json` | 112 images, 127 insect polygons, COCO 1.0 format, annotated by Pankaj Chaubay in CVAT. Licensed CC BY 4.0. |
| `images/` | Not included. The images were supplied by BSBI for the module and are not redistributed. Place them here with the file names used in the annotation file (`00000000.jpg` ...). |
| `splits/` | Created by `paddyseg split`. |

To annotate your own images: create a CVAT project with one label, draw each insect with the polygon tool, then export the task as **COCO 1.0**. The `paddyseg inspect` command will flag any problem in the export before training.
