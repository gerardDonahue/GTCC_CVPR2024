"""
  This code is obtained from the following githubs:
    https://github.com/Maghoumi/pytorch-softdtw-cuda/blob/master/soft_dtw_cuda.py
    https://github.com/vincent-leguen/DILATE/blob/master/loss/soft_dtw.py
"""

import numpy as np
import torch
from numba import jit
from torch.autograd import Function

@jit(nopython = True)
def compute_softdtw(D, gamma):
  B = D.shape[0]
  N = D.shape[1]
  M = D.shape[2]
  R = np.ones((B, N + 2, M + 2)) * np.inf
  R[:, 0, 0] = 0
  for k in range(B):
    for j in range(1, M + 1):
      for i in range(1, N + 1):
        r0 = -R[k, i - 1, j - 1] / gamma
        r1 = -R[k, i - 1, j] / gamma
        r2 = -R[k, i, j - 1] / gamma
        rmax = max(max(r0, r1), r2)
        rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
        softmin = - gamma * (np.log(rsum) + rmax)
        R[k, i, j] = D[k, i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  B = D_.shape[0]
  N = D_.shape[1]
  M = D_.shape[2]
  D = np.zeros((B, N + 2, M + 2))
  E = np.zeros((B, N + 2, M + 2))
  D[:, 1:N + 1, 1:M + 1] = D_
  E[:, -1, -1] = 1
  R[:, : , -1] = -np.inf
  R[:, -1, :] = -np.inf
  R[:, -1, -1] = R[:, -2, -2]
  for k in range(B):
    for j in range(M, 0, -1):
      for i in range(N, 0, -1):
        a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
        b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
        c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
        a = np.exp(a0)
        b = np.exp(b0)
        c = np.exp(c0)
        E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
  return E[:, 1:N + 1, 1:M + 1]

class _SoftDTW(Function):
  @staticmethod
  def forward(ctx, D, gamma):
    dev = D.device
    dtype = D.dtype
    gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
    D_ = D.detach().cpu().numpy()
    g_ = gamma.item()
    R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
    ctx.save_for_backward(D, R, gamma)
    return R[:, -2, -2]
    # modification made on 12/19/2022
    #return R[:, -2, -2], R[:, 1:-1, 1:-1]

  @staticmethod
  def backward(ctx, grad_output):
    dev = grad_output.device
    dtype = grad_output.dtype
    D, R, gamma = ctx.saved_tensors
    D_ = D.detach().cpu().numpy()
    R_ = R.detach().cpu().numpy()
    g_ = gamma.item()
    E = torch.Tensor(compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
    return grad_output.view(-1, 1, 1).expand_as(E) * E, None
    # modification on 12/19/2022
    #return grad_output.view(-1, 1, 1).expand_as(E) * E, grad_output * E, None

# modification on 12/21/2022
class _SoftDTW_r(Function):
  @staticmethod
  def forward(ctx, D, gamma):
    dev = D.device
    dtype = D.dtype
    gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
    D_ = D.detach().cpu().numpy()
    g_ = gamma.item()
    R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
    ctx.save_for_backward(D, R, gamma)
    return R[:, 1:-1, 1:-1]

  @staticmethod
  def backward(ctx, grad_output):
    dev = grad_output.device
    dtype = grad_output.dtype
    D, R, gamma = ctx.saved_tensors
    D_ = D.detach().cpu().numpy()
    R_ = R.detach().cpu().numpy()
    g_ = gamma.item()
    E = torch.Tensor(compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
    return grad_output * E, None

class SoftDTW(torch.nn.Module):
  def __init__(self, gamma=1.0, normalize=False, dtw_dist='euclidean'):
    super(SoftDTW, self).__init__()
    self.normalize = normalize
    self.gamma=gamma
    self.func_dtw = _SoftDTW.apply
    self.func_dtw_r = _SoftDTW_r.apply
    self.dtw_dist = dtw_dist
    self.cossim = torch.nn.CosineSimilarity(dim=3, eps=1e-6)
    self.dist_funcs = {"euclidean": self.calc_distance_matrix,
                       "cosine_dis": self.calc_cosine_dissimilarity}


  def plot_matrix(self, dist_matrix):
    import matplotlib.pyplot as plt

    x_len = len(dist_matrix[0])
    y_len = len(dist_matrix[0, 0])
    fig, ax = plt.subplots(figsize=(40, 40))

    print(dist_matrix.size())

    intersection_matrix = dist_matrix[0, 0:50, 0:50].cpu().numpy() #np.random.randint(0, 10, size=(max_val, max_val))

    ax.matshow(intersection_matrix, cmap=plt.cm.Blues)

    for i in range(50):
        for j in range(50):
            c = intersection_matrix[i, j]
            ax.text(j, i, "%.2f"%(c), va='center', ha='center')
    plt.xlabel("input")
    plt.ylabel("prototype")
    fig.savefig('full_figure.png')
    
  def calc_distance_matrix(self, x, y):
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    dist = torch.pow(x - y, 2).sum(3)
    # self.plot_matrix(dist)
    return dist

  def calc_cosine_dissimilarity(self, x, y):
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    #dist = torch.pow(x - y, 2).sum(3)
    dist = 1 - self.cossim(x, y)
    return dist

  def partial_align(self, D, gamma):
    dev = D.device
    dtype = D.dtype
    gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
    D_ = D.detach().cpu().numpy()
    g_ = gamma.item()
    R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
    return torch.min(R[:, :, -2],dim=-1)

  def forward(self, x, y, is_complete = True, withR = False):
    assert len(x.shape) == len(y.shape)
    squeeze = False
    if len(x.shape) < 3:
      x = x.unsqueeze(0)
      y = y.unsqueeze(0)
      squeeze = True
    if self.normalize:
      assert is_complete == True, 'not implemented for incomplete sequence'
      #D_xy = self.calc_distance_matrix(x, y)
      D_yx = self.dist_funcs[self.dtw_dist](x, y)
      out_xy = self.func_dtw(D_xy, self.gamma)
      #D_xx = self.calc_distance_matrix(x, x)
      D_xx = self.dist_funcs[self.dtw_dist](x, x)
      out_xx = self.func_dtw(D_xx, self.gamma)
      #D_yy = self.calc_distance_matrix(y, y)
      D_yy = self.dist_funcs[self.dtw_dist](y, y)
      out_yy = self.func_dtw(D_yy, self.gamma)
      result = out_xy - 1/2 * (out_xx + out_yy) # distance
    else:
      #D_xy = self.calc_distance_matrix(x, y)
      D_xy = self.dist_funcs[self.dtw_dist](x, y)
      if is_complete:
        #out_xy = self.func_dtw(D_xy, self.gamma)
        # modifications made on 12/15/2022
        #out_xy, R = self.func_dtw(D_xy, self.gamma)
        # modifications made on 12/21/2022
        R = self.func_dtw_r(D_xy, self.gamma)
        out_xy = self.func_dtw(D_xy, self.gamma)
      else:
        out_xy_tuple = self.partial_align(D_xy, self.gamma)
        out_xy = out_xy_tuple[0]
      result = out_xy # discrepancy
    
    
    #return result.squeeze(0) if squeeze else result
    # modifications made on 12/15/2022
    if not withR:
      return result.squeeze(0) if squeeze else result
    else:
      return result.squeeze(0) if squeeze else result, R.squeeze(0) if squeeze else R
