from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent))
import controls as kc
from mix import N_SINK

GROUPS = {'Z': ('stack4pc', 'k4v2pc', 'k3v2pc')}


def hidden_scores(pipe, context_ids, question_ids, capture_kvzip, capture_scorers, visible):
    from transformers import DynamicCache
    cache = DynamicCache()
    press = capture_kvzip(compression_ratio=0.5, kvzip_plus_normalization=True, n_sink=N_SINK)
    with torch.no_grad(), press(pipe.model):
        pipe.model.model(input_ids=context_ids, past_key_values=cache)
    return cache, {'Z': press.captured[:, 0]}


def configure():
    kc.GROUPS = GROUPS
    kc.score_selectors = hidden_scores


if __name__ == '__main__':
    configure()
    sys.exit(kc.main())
