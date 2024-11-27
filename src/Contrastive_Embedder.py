import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from timeit import default_timer
from functools import partial
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling, RobertaForTokenClassification
from otdd.pytorch.distance import DatasetDistance, FeatureCost
from utils import count_params, count_trainable_params, calculate_stats
from task_configs import get_data, get_optimizer_scheduler, set_decoder_trainable
from utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, l2, MMD_loss
from peft import LoraConfig, get_peft_model
from task_configs import get_config
import random
import torch.nn.functional as F
import torch
import argparse
from types import SimpleNamespace

# Alignment Loss
def align_loss(batch_features, labels, alpha=2):
    """
    Alignment loss for batch features and labels.

    Args:
        batch_features (torch.Tensor): Batch of features, shape (batch_size, feature_len).
        labels (torch.Tensor): Labels corresponding to features, shape (batch_size,).
        alpha (int): Power for the distance, default is 2 (squared).

    Returns:
        torch.Tensor: Alignment loss.
    """
    
    print(f"batch_features shape: {batch_features.shape}")
    print(f"labels shape: {labels.shape}")
    labels = labels.squeeze() 
    pairwise_distances = torch.cdist(batch_features, batch_features, p=2)  # Shape: (batch_size, batch_size)
    positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # Shape: (batch_size, batch_size)
    #print("positive mask :",positive_mask)
    positive_mask.fill_diagonal_(False)  # Ignore self-comparisons

    # Compute distances for positive pairs only
    if positive_mask.sum() > 0:  # Avoid empty positive pairs
        positive_distances = pairwise_distances[positive_mask]
        return positive_distances.pow(alpha).mean()
    else:
        return torch.tensor(0.0, device=batch_features.device)

# Uniformity Loss
def uniform_loss(batch_features, t=2, epsilon = 1e-8):
    """
    Uniformity loss to promote diversity in the embedding space.

    Args:
        batch_features (torch.Tensor): Batch of features, shape (batch_size, feature_len).
        t (int): Temperature parameter, default is 2.

    Returns:
        torch.Tensor: Uniformity loss.
    """
    pairwise_distances = torch.pdist(batch_features, p=2).pow(2)  # Pairwise squared Euclidean distances
    exp_distances = (-t * pairwise_distances).exp()  # Apply exponential
    mean_exp = exp_distances.mean()  # Expectation over all pairs
    return (mean_exp + epsilon).log()
# Combined Contrastive Loss
def contrastive_loss(batch_features, labels, alpha=2, t=2, lambda_align=1.0, lambda_uniform=1.0):
    """
    Combined contrastive loss as a sum of alignment and uniformity losses.

    Args:
        batch_features (torch.Tensor): Batch of features, shape (batch_size, feature_len).
        labels (torch.Tensor): Labels corresponding to features, shape (batch_size,).
        alpha (int): Power for alignment loss, default is 2.
        t (int): Temperature parameter for uniformity loss, default is 2.
        lambda_align (float): Weight for alignment loss.
        lambda_uniform (float): Weight for uniformity loss.

    Returns:
        torch.Tensor: Combined contrastive loss.
    """
    align = align_loss(batch_features, labels, alpha)
    uniform = uniform_loss(batch_features, t)
    return lambda_align * align + lambda_uniform * uniform




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

     
class wrapper2D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='base', train_epoch=0, activation=None, target_seq_len=None, drop_out=None, from_scratch=False, warm_init = True, classification = None, train_embedder = False):
        super().__init__()
        self.classification = (not isinstance(output_shape, tuple)) and (output_shape != 1)
        self.output_raw = True
        self.train_embedder = train_embedder
        if (classification != None):
            self.classification = classification

        print("2D classification : ", self.classification)    
        if weight == 'tiny':
            arch_name = "microsoft/swin-tiny-patch4-window7-224"
            embed_dim = 96
            output_dim = 768
            img_size = 224
        elif weight == 'base':
            arch_name = "microsoft/swin-base-patch4-window7-224-in22k"
            embed_dim = 128
            output_dim = 1024
            img_size = 224
            patch_size = 4

        if self.classification:
            modelclass = SwinForImageClassification
        else:
            modelclass = SwinForMaskedImageModeling
            
        self.model = modelclass.from_pretrained(arch_name)
        self.model.config.image_size = img_size
        if drop_out is not None:
            self.model.config.hidden_dropout_prob = drop_out 
            self.model.config.attention_probs_dropout_prob = drop_out

        self.model = modelclass.from_pretrained(arch_name, config=self.model.config) if not from_scratch else modelclass(self.model.config)

        if self.classification:
            self.model.pooler = nn.AdaptiveAvgPool1d(1)
            self.model.classifier = nn.Identity()
            self.predictor = nn.Linear(in_features=output_dim, out_features=output_shape)
        else:
            self.pool_seq_dim = adaptive_pooler(output_shape[1] if isinstance(output_shape, tuple) else 1)
            self.pool = nn.AdaptiveAvgPool2d(input_shape[-2:])
            self.predictor = nn.Sequential(self.pool_seq_dim, self.pool)

        set_grad_state(self.model, False)
        # set_grad_state(self.predictor, False) 

        if use_embedder:
            self.embedder = Embeddings2D(input_shape, patch_size=patch_size, config=self.model.config, embed_dim=embed_dim, img_size=img_size)
            embedder_init(self.model.swin.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)
            self.model.swin.embeddings = self.embedder  


    def forward(self, x):
       
        if self.output_raw:
            if self.classification:
                if self.train_embedder:
                   return self.model.swin.embeddings(x)[0]
                else :   
                   return self.model(x).logits
                
            else:
                if self.train_embedder:
                    return self.model.swin.embeddings(x)[0]
                else:
                    embedding_output, input_dimensions = self.model.swin.embeddings(x)
                    encodder_output = self.model.swin.encoder(embedding_output, input_dimensions)
                    output_affterEncodder =  encodder_output.last_hidden_state
                    return output_affterEncodder
        
        if self.train_predictor:
            if self.classification:    
               return self.predictor(x)
            else: 
                decodder_outout = self.model.decoder(x)
                return self.predictor(decodder_outout)

        x = self.model(x).logits
        return self.predictor(x)


class wrapper1D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='roberta', train_epoch=0, activation=None, target_seq_len=512, drop_out=None, from_scratch=False, warm_init = True):
        super().__init__()

        self.dense = False
        self.output_raw = True
        self.weight = weight
        self.output_shape = output_shape
        
        if isinstance(output_shape, tuple):
            self.dense = True
        if (from_scratch):
            print("randominit model")
        if weight =='swin':
            self.model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k") if not from_scratch else SwinForImageClassification()
            self.model.pooler = nn.AdaptiveAvgPool1d(1)
            self.model.classifier = nn.Identity() 

        else:
            modelname = 'roberta-base' if weight[:7] == 'roberta' else 'bert-base-uncased'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)
            
        print("nomal 1D lora ",count_params(self.model))
        if use_embedder:
            self.embedder = Embeddings1D(input_shape, config=self.model.config, embed_dim=128 if weight == 'swin' else 768, target_seq_len=1024 if weight == 'swin' else target_seq_len, dense=self.dense)
            embedder_init(self.model.swin.embeddings if weight == 'swin' else self.model.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)    
        else:
            self.embedder = nn.Identity()

        if not weight == 'swin': 
            self.model.embeddings = embedder_placeholder()
            if self.dense:
                self.model.pooler = nn.Identity()
                self.predictor = adaptive_pooler(out_channel = output_shape[-2] * self.embedder.stack_num, output_shape=output_shape, dense=True)
            else:
                self.model.pooler = adaptive_pooler()
                self.predictor = nn.Linear(in_features=768, out_features=output_shape)   
        else:
            self.model.swin.embeddings = self.embedder  
            if self.dense:
                self.predictor = adaptive_pooler(out_channel = output_shape[-2] * self.embedder.stack_num)
            else:
                self.predictor = nn.Linear(in_features=1024, out_features=output_shape)  

        if activation == 'sigmoid':
            self.predictor = nn.Sequential(self.predictor, nn.Sigmoid())  

        print("final 1D nomal",count_params(self.model))    
            
        # self.model is body model     
        #set_grad_state(self.model, False)
        #set_grad_state(self.predictor, False)


    def forward(self, x):
        if self.weight == 'swin':
            if self.output_raw:
                return  self.model(x).logits
            # nomal foward 
            x = self.model(x).logits
            return self.predictor(x)
        

        x = self.embedder(x)
        # foward with dense and not dense
        if self.dense:
            x = self.model(inputs_embeds=x)['last_hidden_state']
            if self.output_raw :
                return x
            x = self.predictor(x)
        else:
            x = self.model(inputs_embeds=x)['pooler_output']
            if self.output_raw :
                return x
            x = self.predictor(x)

        if x.shape[1] == 1 and len(x.shape) == 2:
            x = x.squeeze(1)

        return x



class Embeddings2D(nn.Module):

    def __init__(self, input_shape, patch_size=4, embed_dim=96, img_size=224, config=None):
        super().__init__()

        self.resize, self.input_dimensions = transforms.Resize((img_size, img_size)), (img_size, img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patched_dimensions = (self.input_dimensions[0] // self.patch_size[0], self.input_dimensions[1] // self.patch_size[1])
        ks = self.patch_size
        self.projection = nn.Conv2d(input_shape[1], embed_dim, kernel_size=ks, stride=self.patch_size, padding=(ks[0]-self.patch_size[0]) // 2)
        self.norm = nn.LayerNorm(embed_dim)
        num_patches = (self.input_dimensions[1] // self.patch_size[1]) * (self.input_dimensions[0] // self.patch_size[0])
        
        conv_init(self.projection)

        
    def maybe_pad(self, x, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            x = nn.functional.pad(x, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            x = nn.functional.pad(x, pad_values)
        return x


    def forward(self, x, *args, **kwargs):
        x = self.resize(x)
        _, _, height, width = x.shape

        x = self.maybe_pad(x, height, width)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        
        x = self.norm(x)   
        
        return x, self.patched_dimensions


class Embeddings1D(nn.Module):
    def __init__(self, input_shape, embed_dim=768, target_seq_len=64, config=None, dense=False):
        super().__init__()
        self.dense = dense
        self.embed_dim = embed_dim
        self.stack_num = self.get_stack_num(input_shape[-1], target_seq_len)
        self.patched_dimensions = (int(np.sqrt(input_shape[-1] // self.stack_num)), int(np.sqrt(input_shape[-1] // self.stack_num)))
        self.norm = nn.LayerNorm(embed_dim)
        self.padding_idx = 1
        self.position_embeddings = nn.Embedding(target_seq_len, embed_dim, padding_idx=self.padding_idx)

        self.projection = nn.Conv1d(input_shape[1], embed_dim, kernel_size=self.stack_num, stride=self.stack_num)
        conv_init(self.projection)


    def get_stack_num(self, input_len, target_seq_len):
        if self.embed_dim == 768:
            for i in range(1, input_len + 1):
                if input_len % i == 0 and input_len // i <= target_seq_len:
                    break
            return i
        else:
            for i in range(1, input_len + 1):
                root = np.sqrt(input_len // i)
                if input_len % i == 0 and input_len // i <= target_seq_len and int(root + 0.5) ** 2 == (input_len // i):
                    break
            return i


    def forward(self, x=None, inputs_embeds=None, *args, **kwargs):
        if x is None:
            x = inputs_embeds
        b, c, l = x.shape

        x = self.projection(x).transpose(1, 2)
        x = self.norm(x)
            
        position_ids = create_position_ids_from_inputs_embeds(x, self.padding_idx)
        self.ps = self.position_embeddings(position_ids)
        x = x + self.ps

        if self.embed_dim == 768:
            return x
        else:
            return x, self.patched_dimensions



####################################################

def get_tgt_model(args, root, sample_shape, num_classes, loss,lora_rank =1 ,add_loss=False, use_determined=False, context=None, opid=0, mode = 'lora', logging = None, warm_init = True, train_embedder = True):
    
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    if len(sample_shape) == 4:
        IMG_SIZE = 224 if args.weight == 'tiny' or args.weight == 'base' else 196
            
        src_model = wrapper2D(sample_shape, num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out, classification = False, train_embedder= False)
        src_model = src_model.to(args.device).eval()
        src_model.train_embedder = True    
        src_feats = []
        src_ys = []
        for i, data in enumerate(src_train_loader):
            x_, y_ = data 
            x_ = x_.to(args.device)
            x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
            
            out = src_model(x_)
            
            if len(out.shape) > 2:
                out = out.mean(1)
            if (i==1):
                print("shape of source: ", out.shape)
            #src_ys.append(y_.detach().cpu())
            src_ys.append(torch.zeros_like(y_.detach().cpu()))
            src_feats.append(out.detach().cpu())
        src_feats = torch.cat(src_feats, 0)
        src_ys = torch.cat(src_ys, 0).long()
        
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)        
        del src_model, src_train_loader    

    else:
        src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
        del src_train_loader
    print("computing source feature: done")  
    print("src_train_dataset_x_shape: ", next(iter(src_train_dataset))[0].shape) 
    torch.cuda.empty_cache() 
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
        
    if args.infer_label:
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader)
    else:
        num_classes_new = num_classes

    print("src feat shape", src_feats.shape, src_ys.shape, "num classes", num_classes_new) 

    #tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, num_classes_new)

    wrapper_func = wrapper1D if len(sample_shape) == 3 else wrapper2D
    tgt_model = wrapper_func(sample_shape, num_classes,weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
    tgt_model = tgt_model.to(args.device).train()
    
    print("Wrapper_func : ")
    print("all param count:", count_params(tgt_model))
    print("trainabel params count :  ",count_trainable_params(tgt_model))
    args, _, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder')
    tgt_model_optimizer.zero_grad()
    print("get_optimizer : ")
    print("all param count:", count_params(tgt_model))
    print("trainabel params count :  ",count_trainable_params(tgt_model))
    if args.objective == 'otdd-exact':
        score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)
    elif args.objective == 'otdd-gaussian':
        score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=False)
    elif args.objective == 'l2':
        score_func = partial(l2, src_train_dataset=src_train_dataset)
    else:
        score_func = MMD_loss(src_data=src_feats, maxsamples=args.maxsamples)
    
    score = 0
    total_losses, times, embedder_stats = [], [], []
    # Train embeder 
    print("Train embedder with ep = ",args.embedder_epochs)
    print("len of tgt_train_loader : ", len(tgt_train_loader))
    tgt_train_loader_list = list(tgt_train_loader)  # Convert DataLoader to a list
    num_samples = args.maxsamples
    print("num_sample: ", num_samples)
    tgt_model.train_embedder = True
    for ep in range(100):   
        random_batches = random.sample(tgt_train_loader_list, num_samples)
        total_loss = 0    
        time_start = default_timer()
        features = []
        for batch in random_batches:
            if transform is not None:
                     x, y, z = batch
            else:
                     x, y = batch 
            
            x = x.to(args.device)
            out = tgt_model(x)
            if len(out.shape) > 2:
                out = out.mean(1)
          
            features.append(out)
            if len(features) == args.maxsamples :
               print("shape of features: ", len(features))
               features = torch.cat(features, 0)
               embedder_loss = (len(features)/len(random_batches)) * score_func(features)
               embedder_loss.backward()
               print("loss func: ", embedder_loss.item())
               break

        tgt_model.train_embedder = False
        x_batch = []
        y_batch = []
        for batch in random_batches: 
            if transform is not None:
                     x, y, z = batch
            else:
                     x, y = batch 
            
            x = x.to(args.device)
            out = tgt_model(x)
            if len(out.shape) > 2:
                out = out.mean(1)
            x_batch.append(out)    
            y_batch.append(y)
            
            if len(x_batch) == 16:
                print("len of len x_batch: ", len(x_batch))
                x_batch = torch.cat(x_batch, 0)
                y_batch = torch.cat(y_batch, 0).long()
                c_loss = (16/len(random_batches))*contrastive_loss(x_batch,y_batch)
                c_loss.backward()
                print("contrastive loss func: ", c_loss.item())
                break

        tgt_model.train_embedder = True
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


        tgt_model_optimizer.step()
        tgt_model_scheduler.step()
        tgt_model_optimizer.zero_grad()

    del tgt_train_loader
    torch.cuda.empty_cache()
    tgt_model.train_predictor = False
    tgt_model.train_embedder = False
    tgt_model.output_raw = False
    return tgt_model, embedder_stats



def load_by_class(loader, num_classes):
    train_set = loader.dataset
    subsets = {}
    print("before if in load by class")
    if len(train_set.__getitem__(0)) == 3:
        try:
            print("first try")
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            print("first except: ")
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    else:
        try:
            print("second try")
            #subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for target in range(num_classes)}

            for target in range(num_classes):
                print(f"Processing class {target}/{num_classes - 1}")
                subsets[target] = torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target])

            print("Processing complete!")
        except:   
           print("second except :")
           subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for target in range(num_classes)}
            #subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}

      
    loaders = {target: torch.utils.data.DataLoader(subset, batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True) for target, subset in subsets.items()}
    class_weights = {target: len(subset)/len(train_set) for target, subset in subsets.items()}
    
    print("class weights")
    for target, subset in subsets.items():
        print(target, len(subset), len(train_set), len(subset)/len(train_set))

    return loaders, class_weights



def get_tensors(dataset):
    xs, ys, zs = [], [], []
    for i in range(dataset.__len__()):
        data = dataset.__getitem__(i)
        xs.append(np.expand_dims(data[0], 0))
        ys.append(np.expand_dims(data[1], 0))
        if len(data) == 3:
            zs.append(np.expand_dims(data[2], 0))

    xs = torch.from_numpy(np.array(xs)).squeeze(1)
    ys = torch.from_numpy(np.array(ys)).squeeze(1)

    if len(zs) > 0:
        zs = torch.from_numpy(np.array(zs)).squeeze(1)
    else:
        zs = None

    return xs, ys, zs

def infer_labels(loader, k = 10):
    from sklearn.cluster import k_means, MiniBatchKMeans
    
    if hasattr(loader.dataset, 'tensors'):
        X, Y = loader.dataset.tensors[0].cpu(), loader.dataset.tensors[1].cpu().numpy()
        try:
            Z = loader.dataset.tensors[2].cpu()
        except:
            Z = None
    else:
        X, Y, Z = get_tensors(loader.dataset)

    Y = Y.reshape(len(Y), -1)

    if len(Y) <= 10000:
        labeling_fun = lambda Y: torch.LongTensor(k_means(Y, k)[1])
        Y = labeling_fun(Y).unsqueeze(1)
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000).fit(Y)
        Y = torch.LongTensor(kmeans.predict(Y)).unsqueeze(1)

    if Z is None:
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, Z), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k

def Test():
    parser = argparse.ArgumentParser(description='ORCA')
    parser.add_argument('--config', type=str, default=None, help='config file name')
    parser.add_argument('--lora_rank', type= int, default= -1, help='LORA rank')
    parser.add_argument('--mode', type= str, default= 'lora', help='mode for ada or lora')
    parser.add_argument('--embedder_ep', type= int, default= None, help='embedder epoch training')
    parser.add_argument('--save_per_ep', type= int, default= 1, help='save per epoch')
    parser.add_argument('--root_dataset', type= str, default= None, help='[option]path to customize dataset')
    parser.add_argument('--log_folder', type= str, default= None, help='[option]path to log folder')
    parser.add_argument('--warm_init', type= bool, default= True, help='warm init controller')
    args = parser.parse_args()
    lora_rank = args.lora_rank
    embedder_ep = args.embedder_ep
    save_per_ep = args.save_per_ep
    mode = args.mode 
    root_dataset = args.root_dataset
    log_folder = args.log_folder
    warm_init = args.warm_init
    print("current mode: ", mode)
    if args.config is not None:     
        import yaml

        with open(args.config, 'r') as stream:
            #args = AttrDict(yaml.safe_load(stream)['hyperparameters']
            #with open('configs/cifar100.yaml', 'r') as stream:
            config = yaml.safe_load(stream)
            args = SimpleNamespace(**config['hyperparameters'])
            args.experiment_id = lora_rank
            if (embedder_ep != None): 
                args.embedder_epochs = embedder_ep
            if (mode == 'from_scratch'):
                args.experiment_id = -2
            if (args.embedder_epochs > 0):
                args.finetune_method = args.finetune_method + 'orca' + str(args.embedder_epochs)
            
            root = './datasets'
            dims, sample_shape, num_classes, loss, args = get_config(root, args)
            args.device = 'cuda'
            model, embedder_stats = get_tgt_model(args, root, sample_shape, num_classes, loss,lora_rank ,False)

if __name__ == '__main__':
    Test()

