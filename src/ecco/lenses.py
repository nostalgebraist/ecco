import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from tqdm.autonotebook import tqdm, trange
from sklearn.decomposition import sparse_encode

import torch
import ecco
import ecco.lm
import ecco.output
import ecco.torch_util

# TODO: put this in a context manager block
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

def _rms(x):
  return np.sqrt((x**2).mean())

def _rms_relative_err(x, xtrue):
  return _rms(x-xtrue) / _rms(xtrue)

class _LensHeadBase:
    def __init__(self,
                 fn,
                 inv_fn=None,
                 tensor_inputs_and_outputs=True,
                 ):
        self.fn = fn
        self.inv_fn = inv_fn
        self.tensor_inputs_and_outputs = tensor_inputs_and_outputs

        if self.tensor_inputs_and_outputs:
            self._to_input = ecco.torch_util.to_tensor
        else:
            self._to_input = lambda x, device: ecco.torch_util.to_numpy(x)

    def __call__(self, h, h_prev=None, device='cpu', invert=False):
        if h_prev is not None:
            h_input = self._to_input(h, device) - self._to_input(h_prev, device)
        else:
            h_input = self._to_input(h, device)
        if invert:
            if self.inv_fn is None:
                raise ValueError(f"lens lacks an inv_fn")
            out = self.inv_fn(h_input)
        else:
            out = self.fn(h_input)
        out_tensor = ecco.torch_util.to_tensor(out, device)
        return out_tensor

    def invert(self, h_lensed, device='cpu'):
        return self(h_lensed, device=device, invert=True)


class LensHead(_LensHeadBase):
    def __init__(self, fn, inv_fn=None):
        super().__init__(fn, inv_fn=inv_fn, tensor_inputs_and_outputs=True)


class NumpyLensHead(_LensHeadBase):
    def __init__(self, fn, inv_fn=None):
        super().__init__(fn, inv_fn=inv_fn, tensor_inputs_and_outputs=False)


NO_LENS = LensHead(fn=lambda x: x)

def make_layer_norm_lens(output: ecco.output.OutputSeq):
    return LensHead(fn=output.layer_norm)

def make_logit_lens(output: ecco.output.OutputSeq):
    return LensHead(fn=lambda x: output.lm_head(output.layer_norm_f(x)))

def make_spike_lens(lm: ecco.lm.LM,
                    layer_num: int,
                    subtract_mean_from_weights=False,
                    variant="sparse",
                    variant_kwargs={},
                    ):
    for name, p in lm.model.transformer.h.named_parameters():
      if name == f"{layer_num}.mlp.c_proj.weight":
        spike_basis = p
      if name == f"{layer_num}.mlp.c_proj.bias":
        spike_basis_bias = p

    if subtract_mean_from_weights:
        spike_basis = ecco.torch_util.subtract_mean(spike_basis)
        spike_basis_bias = ecco.torch_util.subtract_mean(spike_basis_bias)

    numpy_head = False

    if variant == "sparse":
        numpy_head = True

        spike_basis_np = spike_basis.cpu().detach().numpy()
        spike_basis_bias_np = spike_basis_bias.cpu().detach().numpy()

        norms_for_coder = np.linalg.norm(spike_basis_np, axis=1)
        spike_basis_for_coder = (spike_basis_np.T / norms_for_coder).T

        def _fn(h):
            result = sparse_encode(X=h-spike_basis_bias_np,
                                   dictionary=spike_basis_for_coder,
                                   algorithm=variant_kwargs.get('algorithm', 'lasso_cd'),
                                   alpha=variant_kwargs.get("alpha", 5e-2),
                                   max_iter=variant_kwargs.get("max_iter", 1000),
                                   verbose=variant_kwargs.get("verbose", 0),
                                   )
            result = result/norms_for_coder
            return result
        def _inv_fn(lensed_h):
            return (lensed_h * norms_for_coder).dot(spike_basis_for_coder) + spike_basis_bias_np
    else:
        # TODO: implement pseudoinverse, raw multiply, others?
        raise ValueError(f"variant {repr(variant)}")

    lens_class = NumpyLensHead if numpy_head else LensHead
    return lens_class(fn=_fn, inv_fn=_inv_fn)


def lensed_subblock_states(output: ecco.output.OutputSeq,
                           position=0,
                           lens_head: _LensHeadBase=None,
                           lens_head_on_diff=False,
                           subtract_means=True,
                           do_layer_norm=False,
                           max_layers=None,
                           use_tqdm=False):
    if lens_head is None:
        lens_head = make_layer_norm_lens(output)

    subblock_state_array, names = output.subblock_states(
        position=position,
        subtract_means=subtract_means,
        do_layer_norm=do_layer_norm,
        max_layers=max_layers,
    )

    rows = []
    errs = []

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
        if lens_head.inv_fn is not None:
            h_inv = lens_head.invert(h_lensed)
            reconstruction_err = _rms_relative_err(h_inv, h)
            errs.append(reconstruction_err)

        h_prev = h
        prev_lensed = h_lensed

    rows_np = [ecco.torch_util.to_numpy(row) for row in rows]
    a = np.stack(rows_np, axis=0)

    # TODO: cleanup
    pre_df = a[:, 0, :]  # remove position singleton axis
    df = pd.DataFrame(pre_df, index=names, )

    if lens_head.inv_fn is not None:
        errs_np = [ecco.torch_util.to_numpy(row) for row in errs]
        df["metadata_errs"] = errs_np

    return df


def plot_lensed_subblock_states(states,
                                layer_start=None,
                                layer_end=None,
                                diff=False,
                                clip_percentile=None,
                                cbar=False,
                                ):
    if clip_percentile is None:
        # note: sns "robust" uses 2nd and 98th %iles
        clip_percentile = 99.7 if diff else 98

    to_plot = states.loc[layer_start:layer_end]
    to_plot = to_plot.drop([c for c in to_plot.columns if str(c).startswith("metadata")],
                           axis=1)

    if diff:
      to_plot = to_plot.diff(axis=0).dropna()

    to_plot = to_plot.iloc[::-1, :]

    plt.figure(figsize=(12, to_plot.shape[0]))

    vmin = np.nanpercentile(to_plot, 100-clip_percentile)
    vmax = np.nanpercentile(to_plot, clip_percentile)

    sns.heatmap(to_plot,
                cbar=cbar,
                center=0,
                robust=False,
                vmin=vmin, vmax=vmax,
                cmap='vlag');
    plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

    plt.grid(which='minor', c='k')
    plt.show()
