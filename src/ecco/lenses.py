import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from tqdm.autonotebook import tqdm, trange
from sklearn.decomposition import sparse_encode

import torch
import ecco

# TODO: move
def _to_tensor(x, device):
    return x.to(device) if isinstance(x, torch.Tensor) else torch.as_tensor(x).to(device)

# TODO: move
def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

class _LensHeadBase:
    def __init__(self,
                 fn,
                 tensor_inputs_and_outputs=True,
                 ):
        self.fn = fn
        self.tensor_inputs_and_outputs = tensor_inputs_and_outputs

        if self.tensor_inputs_and_outputs:
            self._to_input = _to_tensor
        else:
            self._to_input = lambda x, device: _to_numpy(x)

    def __call__(self, h, h_prev=None, device='cpu'):
        if h_prev is not None:
            h_input = self._to_input(h, device) - self._to_input(h_prev, device)
        else:
            h_input = self._to_input(h, device)
        out = self.fn(h_input)
        out_tensor = _to_tensor(out, device)
        return out_tensor


class LensHead(_LensHeadBase):
    def __init__(self, fn):
        super().__init__(fn, tensor_inputs_and_outputs=True)


class NumpyLensHead(_LensHeadBase):
    def __init__(self, fn):
        super().__init__(fn, tensor_inputs_and_outputs=False)


NO_LENS = LensHead(fn=lambda x: x)


def lensed_subblock_states(output: ecco.output.OutputSeq,
                           position=0,
                           lens_head: _LensHeadBase=NO_LENS,
                           lens_head_on_diff=False,
                           subtract_means=True,
                           max_layers=None,
                           use_tqdm=False):
    subblock_state_array, names = output.subblock_states(
        position=position,
        subtract_means=subtract_means,
        max_layers=max_layers,
    )

    rows = []

    _iter = zip(names, subblock_state_array)
    if use_tqdm:
        _iter = tqdm(_iter)

    h_prev = None
    prev_lensed = None
    for name, h in _iter:
        head_out = lens_head(h, h_prev=h_prev, device=output.device)
        if lens_head_on_diff and prev_lensed is not None:
            h_lensed = prev_lensed + head_out
        else:
            h_lensed = head_out

        rows.append(h_lensed)

        h_prev = h
        prev_lensed = h_lensed

    rows_np = [_to_numpy(row) for row in rows]
    a = np.stack(rows_np, axis=0)

    # TODO: cleanup
    pre_df = a[:, 0, :]  # remove position singleton axis
    df = pd.DataFrame(pre_df, index=names, )
    return df


def plot_lensed_subblock_states(states,
                                layer_start=None,
                                layer_end=None,
                                diff=False,
                                clip_percentile=None,
                                ):
    if clip_percentile is None:
        # note: sns "robust" uses 2nd and 98th %iles
        clip_percentile = 99.7 if diff else 98

    to_plot = states.loc[layer_start:layer_end]

    if diff:
      to_plot = to_plot.diff(axis=0).dropna()

    to_plot = to_plot.iloc[::-1, :]

    plt.figure(figsize=(12, to_plot.shape[0]))

    vmin = np.nanpercentile(to_plot, 100-clip_percentile)
    vmax = np.nanpercentile(to_plot, clip_percentile)

    sns.heatmap(to_plot,
                cbar=False,
                center=0,
                robust=False,
                vmin=vmin, vmax=vmax,
                cmap='vlag');
    plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

    plt.grid(which='minor', c='k')
    plt.show()
