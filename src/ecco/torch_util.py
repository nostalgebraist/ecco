import torch


def to_tensor(x, device):
    return x.to(device) if isinstance(x, torch.Tensor) else torch.as_tensor(x).to(device)

def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def subtract_mean(t: torch.Tensor, dim=-1):
  return (t.T-t.mean(dim=dim).T).T
