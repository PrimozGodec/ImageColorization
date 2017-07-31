import argparse

# from src.models import reg_ful_model
import src.models.reg_ful_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--model', type=str, default="reg_ful_model", help="Choose name of the model you want to test")

    args = parser.parse_args()

    # import from user selected model
    imported_model = __import__('src.models.' + args.model, fromlist=[''])
    model = imported_model.model()

    # load weights
    model.load_weights(imported_model.weights)

    # color images
    imported_model.color_fun(model, imported_model.name)

