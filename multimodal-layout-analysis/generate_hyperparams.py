import argparse
import yaml
import itertools

from pathlib import Path
from tqdm import tqdm


def main(args):
    if not args.yaml_hyperparams_path.exists():
        print(f'Could not find hyperparams file {args.yaml_hyperparams_path}')
        exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.yaml_hyperparams_path, 'r', encoding='utf-8') as file:
        hyperparams = yaml.safe_load(file)

    hyperparam_lists_keys = []
    hyperparam_lists_values = []
    hyperparam_values = []
    for hyperparam in hyperparams.items():
        if isinstance(hyperparam[1], list):
            hyperparam_lists_keys.append(hyperparam[0])
            hyperparam_lists_values.append(hyperparam[1])
        else:
            hyperparam_values.append(hyperparam)

    for idx, combination in enumerate(tqdm(itertools.product(*hyperparam_lists_values))):
        config = {}

        for hyperparam_idx, hyperparam_value in enumerate(combination):
            config[hyperparam_lists_keys[hyperparam_idx]] = hyperparam_value

        for value in hyperparam_values:
            config[value[0]] = value[1]

        with open(args.output_dir / f'config_{idx+1}.yaml', 'w', encoding='utf-8') as file:
            yaml.dump(config, file)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--yaml-hyperparams-path', type=Path, required=True,
                           default=None)
    argparser.add_argument('--output-dir', type=Path, required=True,
                           default=None)

    args = argparser.parse_args()
    main(args)
