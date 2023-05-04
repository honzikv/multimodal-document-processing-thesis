import argparse
import tqdm
import pandas as pd
from time import sleep

import wandb

RELEVANT_COL = ['val_f1_epoch', 'val_acc_epoch', 'val_precision_epoch', 'val_recall_epoch', 'val_loss_epoch']


def make_summary(run, best_metric='val_f1_epoch') -> bool:
    """
    Make a summary for the given run.
    Args:
        run: The run to make a summary for.
        best_metric: The metric to use to find the best row.

    Returns:
        True if the summary was made, False otherwise.
    """
    history = run.scan_history(keys=RELEVANT_COL)
    history = pd.DataFrame([row for row in history])
    for col in RELEVANT_COL:
        if col not in history:
            print(f'Skipping run: {run.id} because it does not have column: {col}')
            return False

    if history[best_metric].isnull().all():
        print(f'Skipping run: {run.id} because it does not have any non-null values for column: {best_metric}')
        return False

    best_row = history.loc[history[best_metric].idxmax()]

    for col in RELEVANT_COL:
        run.summary[f'best_{col}'] = best_row[col]

    return True


def main(args):
    entity, project = args.entity, args.project
    api = wandb.Api()

    runs = tqdm.tqdm(api.runs(f'{entity}/{project}'))
    for run in runs:
        save = make_summary(run)
        # sleep(.05)

        if save:
            run.update()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--entity', type=str, required=True)
    argparser.add_argument('--project', type=str, required=True)

    args = argparser.parse_args()

    main(args)
