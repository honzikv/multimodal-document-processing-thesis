import argparse

from ultralytics import YOLO


def save_metrics(metrics, model_path):
    with open(model_path, 'w') as f:
        f.write(metrics)


def run_validation(model_path):
    model = YOLO(model_path)
    metrics = model.val()
    print(metrics)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model-path', type=str, help='Path to the model', required=True)
    args = argparser.parse_args()
    run_validation(args.model_path)
