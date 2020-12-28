import numpy as np
from tqdm.autonotebook import tqdm, trange
from sklearn.decomposition import sparse_encode

import ecco


def lensed_subblock_states(output: ecco.output.OutputSeq,
                           position=None,
                           lens_head=None,
                           lens_head_is_np=False,
                           lens_head_on_diff=False,
                           subtract_means=True,
                           max_layers=None,
                           use_tqdm=False):
    def _to_user(tensor, prev=None, prev_for_user=None):
        if lens_head is not None:
            if lens_head_is_np:
              if lens_head_on_diff and prev is not None and prev_for_user is not None:
                prev_ = prev.detach().cpu().numpy()
                this_ = tensor.detach().cpu().numpy()
                h_for_user = prev_for_user + lens_head(this_-prev_)
              else:
                h_for_user = lens_head(tensor.detach().cpu().numpy())
            else:
              if lens_head_on_diff and prev is not None and prev_for_user is not None:
                diff = lens_head(output.to(tensor)-output.to(prev))
                h_for_user = prev_for_user + diff.detach().cpu().numpy()
              else:
                h_for_user = lens_head(output.to(tensor)).detach().cpu().numpy()
        else:
            h_for_user = tensor.numpy()
        return h_for_user

    subblock_state_array, names = output.subblock_states(
        position=position,
        subtract_means=subtract_means,
        max_layers=max_layers,
    )

    rows = []

    _iter = zip(names, subblock_state_array)
    if use_tqdm:
        _iter = tqdm(_iter)

    prev = None
    prev_for_user = None
    for name, h in _iter:
        h_for_user = _to_user(h, prev=prev, prev_for_user=prev_for_user)
        rows.append(h_for_user)

        prev = h
        prev_for_user = h_for_user

    a = np.stack(rows, axis=0)
    return a
