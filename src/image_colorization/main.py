import argparse

from src.utils.image_utils import get_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--model',
                        type=str,
                        default="reg_full_model",
                        help="Choose name of the model you want to test",
                        choices={"reg_full_model", "reg_full_vgg_model", "reg_part_model",
                                 "class_weights_model", "class_wo_weights_model"})

    args = parser.parse_args()

    # import from user selected model
    imported_model = __import__('src.models.' + args.model, fromlist=[''])
    model = imported_model.model()

    # load weights
    model.load_weights(get_weights(imported_model.weights))

    # color images
    imported_model.color_fun(model)
