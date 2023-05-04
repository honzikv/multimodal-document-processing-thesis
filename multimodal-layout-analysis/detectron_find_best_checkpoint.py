import argparse
import json
import shutil
import tqdm
from pathlib import Path

# This script is a simple utility to find the best map50:95 checkpoint in a finetuned detectron2 model directory

DEFAULT_MODEL_FILE = 'model_best.pth'


def main(args):
    if args.output_file_path is None:
        print(f'No output file path specified - using default: {DEFAULT_MODEL_FILE}')
        args.output_file_path = args.model_path / DEFAULT_MODEL_FILE
        print(f'Model will be saved to: {args.output_file_path}')

    metrics_file_path = args.model_path / 'metrics.json'
    if not metrics_file_path.exists():
        print(f'Metrics file {metrics_file_path} does not exist - aborting')
        exit(1)

    if args.output_file_path.exists():
        print(f'Output file path {args.output_file_path} already exists - deleting it')
        args.output_file_path.unlink()

    eval_logs = []
    with open(metrics_file_path, 'r', encoding='utf-8') as file:
        print(f'Processing log file')
        for line in tqdm.tqdm(file.readlines()):
            if line.startswith('{"bbox/AP"'):
                eval_logs.append(line)

    best_ckpt = 'model_final.pth'
    best_map50_95 = 0.0
    best_ckpt_log = None
    print(f'Processing eval logs')

    for log in tqdm.tqdm(eval_logs):
        data = json.loads(log)
        if data['segm/AP'] > best_map50_95:
            best_map50_95 = data['segm/AP']
            best_ckpt = f'model_{str(data["iteration"]).zfill(7)}.pth'
            best_ckpt_log = log

    print(f'Best checkpoint: {best_ckpt}')
    print(best_ckpt_log)
    shutil.copy(args.model_path / best_ckpt, args.output_file_path)

    with open(args.model_path / 'best_ckpt_log.json', 'w', encoding='utf-8') as file:
        file.write(best_ckpt_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=Path, required=True,
                        help='Path to the model - must contain metrics.json and checkpoints')
    parser.add_argument('--output-file-path', type=Path, required=False, default=None,
                        help='Path to the output file to save the best checkpoint to. '
                             'Defaults to best.pth in the model path')

    args = parser.parse_args()
    main(args)
