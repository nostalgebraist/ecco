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

    def _at_position(state):
        if position is not None:
            return state[position:position+1, :]
        return state

    rows = []
    names = []

    if max_layers is None:
      max_layers = len(output.hidden_states)

    _iter = list(range(max_layers))
    if use_tqdm:
      _iter = trange(max_layers)
    for lix in _iter:
      h = _at_position(output.hidden_states[lix])
      if subtract_means:
          h = _subtract_mean(h)
      if lix == 0:
        rows.append(_to_user(h))
        names.append(f"h{lix}")

      if lix < len(output.attn_outs):
        h_plus_attn = h + _at_position(output.attn_outs[lix])
        if subtract_means:
          h_plus_attn = _subtract_mean(h_plus_attn)
        rows.append(_to_user(h_plus_attn, prev=h, prev_for_user=rows[-1]))
        names.append(f"h{lix}+attn{lix+1}")

        h_plus_attn_mlp = h + _at_position(output.attn_outs[lix]) + _at_position(output.mlp_outs[lix])
        if subtract_means:
          h_plus_attn_mlp = _subtract_mean(h_plus_attn_mlp)
        rows.append(_to_user(h_plus_attn_mlp, prev=h_plus_attn, prev_for_user=rows[-1]))
        names.append(f"h{lix+1}")

    a = np.stack(rows, axis=0)
    return a, names
