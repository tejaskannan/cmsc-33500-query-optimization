import pandas as pd
import time
import os
import numpy as np
from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from typing import Dict, Any, List, Tuple
from itertools import permutations

from load_dataset import load
from optimizers import get_optimizer
from merge_dataframes import merge_frames, MergeFrame



def get_queries(query_file: RichPath, dfs: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    query_descriptions = query_file.read_by_file_suffix()

    queries: List[Dict[str, Any]] = []
    for description in query_descriptions:
        on: List[MergeFrame] = []
        for frame, left, right in zip(description['frames'][1:], description['left'], description['right']):
            on.append(MergeFrame(frame=dfs[frame], left_on=left, right_on=right))
    
        query = {
            'from': dfs[description['frames'][0]],
            'on': on
        }
        queries.append(query)

    return queries


def profile(queries: List[Dict[str, Any]], n_trials: int = 100) -> Tuple[int, np.array]:
    """
    Profiles the queries to determine the best one
    """
    times = np.zeros(shape=len(queries))
    for t in range(n_trials):
        for i, q in enumerate(queries):
            start = time.time()
            df = merge_frames(q['from'], q['on'], options=[['home', 'away']])
            elapsed = time.time() - start
            if t > 0:
                times[i] += elapsed

    avg_times = times / (n_trials + 1)
    return np.argmin(avg_times), avg_times


def run_experiment(queries: List[Dict[str, Any]], best_index: int, n_trials: int, dfs: Dict[str, pd.DataFrame], dataset_params: Dict[str, Any], optimizer_params: Dict[str, Any], output_file: RichPath):

    times: List[float] = []
    total_time: float = 0.0
    best_count: float = 0.0
    avg_times: List[float] = []
    best_frac: List[float] = []
    selected_index: List[int] = []

    optimizer = get_optimizer(optimizer_params['name'], queries, **optimizer_params['params'])
    for t in range(1, n_trials+1):
        start = time.time()
        query_index, q = optimizer.get_query(time=t)
        df = merge_frames(q['from'], q['on'], options=[['home', 'away']])
        elapsed = time.time() - start
        optimizer.update(query_index, reward=-elapsed)

        df.copy()

        # Avoid time measurements on the first iteration due to caching
        selected_index.append(query_index)
        if t > 1:
            times.append(elapsed)
            total_time += elapsed
            avg_times.append(total_time / (t - 1))
            best_count += float(query_index == best_index)
            best_frac.append(best_count / (t - 1))

        if t % 100 == 0:
            print(f'Completed {t} trials')

    # Collect and write the metrics
    metrics = dict(times=times, avg_times=avg_times, best_frac=best_frac, selected_index=selected_index)
    output_file.save_as_compressed_file(metrics)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n-trials', type=int, required=True)
    parser.add_argument('--optimizer-params', type=str, nargs='+')
    parser.add_argument('--dataset-params', type=str, required=True)
    args = parser.parse_args()

    dataset_file = RichPath.create(args.dataset_params)
    assert dataset_file.exists(), f'The file {dataset_file} does not exist.'
    dataset_params = dataset_file.read_by_file_suffix()

    print('Loading dataframes...')
    dfs: Dict[str, pd.DataFrame] = dict()
    for data_file in dataset_params['files']:
        dfs[data_file['name']] = pd.read_csv(os.path.join(dataset_params['base'], data_file['path']))
    print('Loaded.')

    queries_file = RichPath.create(dataset_params['queries'])
    assert queries_file.exists(), f'The file {queries_file} does not exist.'
    queries = get_queries(queries_file, dfs)

    print('Profiling queries...')
    best_index, avg_times = profile(queries, n_trials=25)
    profile_results_file = RichPath.create('results/profile.pkl.gz')
    profile_results_file.save_as_compressed_file(avg_times)
    print('Finished Profiling.')

    for optimizer_params_file in args.optimizer_params:
        optimizer_file = RichPath.create(optimizer_params_file)
        assert optimizer_file.exists(), f'The file {optimizer_file} does not exist.'
        optimizer_params = optimizer_file.read_by_file_suffix()
        
        run_experiment(queries, best_index, args.n_trials, dfs, dataset_params, optimizer_params, RichPath.create(optimizer_params['output_file']))

        print(f'Finished {optimizer_params_file}')
