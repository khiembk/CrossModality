from otdd.pytorch.distance import DatasetDistance
import torch
from task_configs import get_data
from embedder import load_by_class,wrapper1D,wrapper1DLORA,wrapper2D, wrapper2DLORA
import numpy as np

# Load data
root = './datasets'
src_dataset = 'text'
src_batch_size = 64
src_valid_split = False

tgt_dataset = 'ECG'
tgt_batch_size = 64
def otdd(feats, ys=None, src_train_dataset=None, exact=True):
    ys = torch.zeros(len(feats)) if ys is None else ys

    if not torch.is_tensor(feats):
        feats = torch.from_numpy(feats).to('cpu')
        ys = torch.from_numpy(ys).long().to('cpu')

    dataset = torch.utils.data.TensorDataset(feats, ys)

    dist = DatasetDistance(src_train_dataset, dataset,
                                    inner_ot_method = 'exact' if exact else 'gaussian_approx',
                                    debiased_loss = True, inner_ot_debiased=True,
                                    p = 2, inner_ot_p=2, entreg = 1e-1, ignore_target_labels = False,
                                    device=feats.device, load_prev_dyy1=None)
                
    d = dist.distance(maxsamples = len(src_train_dataset))
    return d
#def get_data(root, dataset, batch_size, valid_split, maxsize=None, get_shape=False):

src_train_loader, _, _, _, _, _, _ = get_data(root, src_dataset,src_batch_size , src_valid_split)

tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, tgt_dataset,tgt_batch_size , src_valid_split)
transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None


src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
tgt_class = 4 
tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, tgt_class)

print("src feat shape", src_feats.shape, src_ys.shape)

wrapper_func = wrapper1D 
dims, sample_shape, num_classes = 1, (1, 1, 1000), 4
tgt_model = wrapper_func(sample_shape, num_classes, 'roberta', train_epoch= 60, activation= None, target_seq_len= 512, from_scratch= False)
tgt_model = tgt_model.to('cuda').train()


total_loss = 0
for i in np.random.permutation(tgt_class):
            feats = []
            datanum = 0

            for j, data in enumerate(tgt_train_loaders[i]):
                
                if transform is not None:
                    x, y, z = data
                else:
                    x, y = data 
                
                x = x.to('cuda')
                out = tgt_model(x)
                feats.append(out)
                
                datanum += x.shape[0]
                
                if datanum > 1000: break

            feats = torch.cat(feats, 0).mean(1)
            print("target shape: ", feats.shape)
            if feats.shape[0] > 1:
                loss = tgt_class_weights[i] * otdd(feats=feats, src_train_dataset= src_train_dataset)
                print("loss at class:",loss)
                total_loss += loss.item()

print(f'OTDD_loss = ', total_loss)
