import pandas as pd
import re
import numpy as np
from collections import namedtuple
from typing import List


MergeFrame = namedtuple('MergeFrame', ['frame', 'left_on', 'right_on'])


def field_format(field: str, options: List[str], index: int) -> str:
    regex = '.{' + str(index) + '}.*'
    match = re.match(regex, field)
    if match is None:
        return field
    choice = np.random.choice(options)
    return field.format(choice)


def merge_frames(df: pd.DataFrame, to_merge: List[MergeFrame], options: List[List[str]] = []):
    result = df
    for frame in to_merge:
        left_on = frame.left_on
        for i, o in enumerate(options):
            left_on = field_format(left_on, o, i)

        right_on = frame.right_on
        for i, o in enumerate(options):
            right_on = field_format(right_on, o, i)

        result = result.merge(right=frame.frame, left_on=left_on, right_on=right_on, how='inner')
    return result
