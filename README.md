Image and video colorizer
=========================

Image and video colorizer is package for automatic image and video colorization. 
Models are allready trained 

## Instalation

% todo

## Image colorization

For automatic image colorizing follow those steps:

1.  Copy images into `/data/image/original` directory

2.  Run `main.py` script from `src/image_colorization/` directory.

        python -m src.image_colorization.main --model <model name>
     
    Parameter `--method` is optional, if not present `reg_full_model` is default.
    It can be choose from this list:

    * `reg_full_model` (default)
    * `reg_full_vgg_model`
    * `reg_part_model`
    * `class_weights_model`
    * `class_wo_weights_model`

3. You can find colored images in `/data/image/colorized` directory.

## Video colorization

For automatic image colorizing follow those steps:

1.  Copy images into `/data/video/original` directory

2.  Run `video_colorizer.py` script from `src/video_colorization/` directory.

        python -m src.video_colorization.video_colorizer
     
    Video colorizer is always using `reg_full_model`.

3. You can find colored videos in `/data/video/colorized` directory.