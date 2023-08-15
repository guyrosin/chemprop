import json
import os
from typing import Dict

import numpy as np

from chemprop.args import TrainArgs
from chemprop.constants import TRAIN_LOGGER_NAME
from .run_training import prepare_for_training, run_training
from chemprop.utils import  makedirs, multitask_mean, timeit


@timeit(logger_name=TRAIN_LOGGER_NAME)
def train_only(args: TrainArgs) -> Dict[str, float]:
    """
    Parses Chemprop training arguments and trains a Chemprop model.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :return: A dictionary mapping each metric in :code:`args.metrics` to its average score.
    """

    info, args.seed, args.save_dir, data, logger = prepare_for_training(args)
    makedirs(args.save_dir)
    data.reset_features_and_targets()

    # If resuming experiment, load results from trained models
    test_scores_path = os.path.join(args.save_dir, "test_scores.json")
    if args.resume_experiment and os.path.exists(test_scores_path):
        print("Loading scores")
        with open(test_scores_path) as f:
            model_scores = json.load(f)
    # Otherwise, train the models
    else:
        model_scores = run_training(args, data, logger)

    # Calculate the average score per metric (across tasks and models)
    metric_to_score = {}
    for metric, scores in model_scores.items():
        avg_multitask_score = multitask_mean(np.array(scores), metric)
        avg_multitask_ensemble_score = np.mean(avg_multitask_score)
        metric_to_score[metric] = avg_multitask_ensemble_score
        
    return metric_to_score
