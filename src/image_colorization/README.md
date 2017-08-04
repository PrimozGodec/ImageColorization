This directory contains script to color images.

You can color the images with running `main.py` script from root of the project.

    python -m src.image_colorization.main --method <name of method>

Parameter `--method` is optional, if not present `reg_full_model` is default.
Method can be choose from this list:

* `reg_full_model`
* `reg_full_vgg_model`
* `reg_part_model`
* `class_weights_model`
* `class_wo_weights_model`
