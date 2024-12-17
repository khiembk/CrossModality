import time
import math

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, transpose


def create_orthonormal_matrix(A):
    # returns an orthonormal matrix (square) of size (min(A.shape), min(A.shape))
    Q, R = torch.qr(A)
    return Q


def get_target_modules_list(model, target_modules):
    target_names = []
    for n, _ in model.named_modules():
        if any(t in n for t in target_modules):
            target_names.append(n)
    return target_names


def replace_svft_with_fused_linear(model, target_modules_list):
    print("Replacing SVFT layers with new Linear layers")

    # filter out svft layer
    target_modules_list = [l for l in target_modules_list if "svft_layer" not in l]
    #print(target_modules_list[0])
    for target_path in tqdm(reversed(target_modules_list), total=len(target_modules_list)):
        #print("target layer: ", target_path, flush=True)
        parent_path = target_path[: target_path.rfind(".")] if "." in target_path else ""
        target_name = target_path.split(".")[-1]
        parent = model.get_submodule(parent_path) if parent_path else model
        target = model.get_submodule(target_path)
        in_dim = target.svft_layer.v.shape[1]
        out_dim = target.svft_layer.u.shape[0]
        if target.bias is None:
            lin = torch.nn.Linear(in_dim, out_dim, bias=False)
        else:
            lin = torch.nn.Linear(in_dim, out_dim, bias=True)
            lin.bias.data = target.bias.data
        lin.weight.data = target.merge_and_unload()
        parent.__setattr__(target_name, lin)


def create_and_replace_modules(model, target_modules_list, create_fn):
    print("Replacing Linear layers with SVFT layers")

    for target_path in tqdm(reversed(target_modules_list), total=len(target_modules_list)):
        #print("target layer: ", target_path, flush=True)
        parent_path = target_path[: target_path.rfind(".")] if "." in target_path else ""
        target_name = target_path.split(".")[-1]
        parent = model.get_submodule(parent_path) if parent_path else model
        target = model.get_submodule(target_path)
        parent.__setattr__(target_name, create_fn(target))


class SVFTLayer(nn.Module):
    def __init__(self, u, s, v, off_diag, pattern="banded", rank=None, fill_orthonormal=False):

        """
        @inputs:
            u: torch.Tensor. Left singular vectors of pre-trained weight matrix
            s: torch.Tensor. Singular values of pre-trained weight matrix
            v: torch.Tensor. Right singular vectors of pre-trained weight matrix
            off_diag: int. Total off-diagonals to be used to populate matrix M (as referred in main paper)
            pattern: str. Choices: "banded", "random", "top_k". Using "banded" with off_diag=1 simulates SVFT-plain
            rank: int. Constraints how many singular vectors and values to use.
            fill_orthonormal: bool. To determine if random orthonormal basis should be used
        """

        super().__init__()

        self.off_diag = off_diag
        rank = s.shape[0] if rank is None else min(s.shape[0], rank)
        self.n = rank
        diff_rank = s.shape[0] - rank

        if fill_orthonormal:
            Q_u = torch.randn_like(u).to(s.device)
            torch.nn.init.orthogonal_(Q_u)
            Q_v = torch.randn_like(v).to(s.device)
            torch.nn.init.orthogonal_(Q_v)

            u = torch.cat([u[:, :rank], Q_u[:, :diff_rank]], dim=1)
            v = torch.cat([v[:rank, :], Q_v[:diff_rank, :]], dim=0)
            s = torch.cat([s[:rank], torch.zeros(diff_rank).to(s.device)], dim=0)
            self.n = s.shape[0]

        else:
            s = s[:rank]
            u = u[:, :rank]
            v = v[:rank, :]

        self.u = nn.Parameter(u.clone().detach().contiguous(), requires_grad=False)

        s_pre = s.cpu().detach().clone().contiguous()

        if s_pre.ndimension() == 1:
            self.s_pre_edge_index = torch.sparse.spdiags(s_pre, torch.LongTensor([0]), (self.n, self.n)).coalesce().indices()
        elif s_pre.ndimension() == 2:
            self.s_pre_edge_index = torch.sparse.spdiags(s_pre, torch.LongTensor([0]), (self.n, self.n)).coalesce().indices()
        else:
            print("s size: ", s.shape)
            s_pre = s_pre.view(-1)  
            self.s_pre_edge_index = torch.sparse.spdiags(s_pre, torch.LongTensor([0]), (self.n, self.n)).coalesce().indices()
        self.s_pre = nn.Parameter(s_pre, requires_grad=False)
        
        if pattern=="banded":  
            diags = 2*self.off_diag + 1
            offsets_positive = torch.arange(0, self.off_diag+1)
            offsets_negative = torch.arange(-1, -self.off_diag-1, -1)
            self.offsets  = torch.cat([offsets_positive, offsets_negative])
            self.s_edge_index = torch.sparse.spdiags(torch.randn([diags, self.n]), self.offsets, (self.n, self.n)).coalesce().indices()
            self.s = torch.nn.Parameter(torch.zeros(self.s_edge_index.shape[1]), requires_grad=True)

        elif pattern=="random":
            print("Random pattern")
            k = self.n*(2*self.off_diag+1) - self.off_diag*(self.off_diag+1)
            rows = torch.randint(0, self.n, (k,))
            cols = torch.randint(0, self.n, (k,))
            self.s_edge_index = torch.stack([rows, cols])
            self.s = torch.nn.Parameter(torch.zeros(k), requires_grad=True)

        elif pattern=="top_k":

            if u.shape == v.shape:
                coeffs = u@v.T
            else:
                coeffs = u if u.shape[0]==u.shape[1] else v

            k = self.n*(2*self.off_diag+1) - self.off_diag*(self.off_diag+1)
            # Flatten the tensor to 1D
            flattened_tensor = coeffs.contiguous().view(-1)
            _, top_indices_flat = torch.topk(flattened_tensor, k)
            num_rows, num_cols = coeffs.size()
            rows = top_indices_flat // num_cols
            cols = top_indices_flat % num_cols
            self.s_edge_index = torch.stack([rows, cols])
            self.s = torch.nn.Parameter(torch.zeros(k), requires_grad=True)
       
        torch.nn.init.kaiming_normal_(self.s[None, :])
        self.s.squeeze()

        self.register_buffer('s_pre_row', self.s_pre_edge_index[0])
        self.register_buffer('s_pre_col', self.s_pre_edge_index[1])
        self.register_buffer('s_row', self.s_edge_index[0])
        self.register_buffer('s_col', self.s_edge_index[1])

        self.gate = nn.Parameter(torch.tensor([0.], dtype=torch.float32), requires_grad=True)

        self.v = nn.Parameter(v.clone().detach().contiguous(), requires_grad=False) 
        

    def forward(self, x):
        x  = x @ self.get_weights() 
        return x

    def set_s_with_pair(self, pair):
        pair_tensor = torch.tensor(pair, dtype=torch.long).T  
        self.s_edge_index = pair_tensor
        self.s = torch.nn.Parameter(torch.zeros(pair_tensor.shape[1]), requires_grad=True)


    def get_weights(self):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # s = SparseTensor(row=self.s_row, col=self.s_col, value=self.s * F.sigmoid(self.gate)).to(device)
        # s_pre = SparseTensor(row=self.s_pre_row, col=self.s_pre_col, value=self.s_pre).to(device)
        s = SparseTensor(row=self.s_row, col=self.s_col, value=self.s*F.sigmoid(self.gate))
        s_pre = SparseTensor(row=self.s_pre_row, col=self.s_pre_col, value=self.s_pre)
        del_s = s_pre + s 

        weight = (del_s @ self.v).T
        weight = weight @ self.u.T
        return weight
    

    def merge_and_unload(self):
        return self.get_weights().T.contiguous()
    
    def compute_score(self, grad_W, i, j):
     
        assert isinstance(grad_W, torch.Tensor), "grad_W_t must be a torch.Tensor"
        u_i = self.u[:, i]  # i-th column of u
        v_j = self.v[j, :]  # j-th row of v

    # Compute (∂L/∂W_t)^T · u_i
        grad_transpose_u_i = grad_W.T @ u_i

    # Compute v_j^T · result
        score = v_j @ grad_transpose_u_i

        return score*score
    
   
class LinearWithSVFT(nn.Module):

    def __init__(self, linear, off_diag, pattern="banded", rank=None, fill_orthonormal=False):
        """
        @inputs:
                linear: torch.Tensor. Linear Layer that has to adapted
                off_diag: int. total number off diagonals to be used if pattern is 'banded' 
                          for remaining patterns, equivalent number of learnable parameters are learnt
                rank: SVD rank 
                fill_orthonormal: bool. To determine if random orthonormal basis should be used
        """
        
        super().__init__()

        self.bias = linear.bias
        
        # since linear.weight is on GPU, computing SVD will be significantly faster
        svd = torch.linalg.svd(linear.weight, full_matrices=False)
        assert svd[1].ndimension()==1  
        # print("s1 size is: ",svd[1].ndimension())
        self.svft_layer = SVFTLayer(svd[0], 
                                    svd[1], 
                                    svd[2], 
                                    off_diag=off_diag, 
                                    pattern=pattern, 
                                    rank=rank, 
                                    fill_orthonormal=fill_orthonormal)
        
        self.m = svd[0].shape[0]
        self.n = svd[2].shape[2]
        self.grad_W = None

    def forward(self, x):
        if self.bias is not None:
            return self.svft_layer(x) + self.bias

        else:
            return self.svft_layer(x)
        
    def merge_and_unload(self):
        return self.svft_layer.merge_and_unload()
    
    def get_score(self, i,j):
        return self.svft_layer.compute_score(self.grad_W,i,j)
    
    def get_sorted_list_score(self):
        score_list = []
        for i in range(self.m):
            for j in range(self.n):
                score_list.append(self.get_score(i,j))

        sorted_scores = sorted(score_list, reverse=True)
        return sorted_scores        

    def get_pair_with_thresould(self, thresould):
        pair_list = []
        for i in range(self.m):
            for j in range(self.n):
                if self.get_score(i,j) >= thresould:
                    pair_list.append((i,j))

        return pair_list
    
    def reconstruct_layer_threshould(self,thresould):
        pair = self.get_pair_with_thresould(thresould)
        self.svft_layer.set_s_with_pair(pair=pair)

    def register_gradient_hook(self):
        """
        Registers a hook to capture the gradient of the effective weights W_t.
        """
        def hook_function(grad):
            # Capture the gradient of the effective weight W_t
            self.grad_W = grad.clone()

        # Register the hook on the dynamically generated weights
        W = self.svft_layer.get_weights()  # Dynamically generated effective weight
        W.retain_grad()  # Ensure gradients are retained for the tensor
        W.register_hook(hook_function)
