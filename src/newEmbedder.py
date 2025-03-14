import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from timeit import default_timer
from functools import partial
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling, RobertaForTokenClassification
from otdd.pytorch.distance import DatasetDistance, FeatureCost
import torch.optim as optim
from math import log
from task_configs import get_data, get_optimizer_scheduler
from utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, l2, MMD_loss
import copy, tqdm
from utils import count_params, count_trainable_params, calculate_stats
from TotalVarianceDistance import OptimalTV,estimate_tv_distance_hist

def total_variance_distance(p, q):
    """
    Compute the Total Variance Distance (TVD) between two distributions p and q.

    Args:
        p (torch.Tensor): Probability distribution p.
        q (torch.Tensor): Probability distribution q.

    Returns:
        torch.Tensor: The Total Variance Distance between p and q.
    """
    return 0.5 * torch.sum(torch.abs(p - q))

def compute_distribution(z, bins=10, range=(0, 1)):
    """
    Compute the probability distribution of z using a histogram.

    Args:
        z (torch.Tensor): Embedded features.
        bins (int): Number of bins for the histogram.
        range (tuple): Range of the histogram.

    Returns:
        torch.Tensor: Probability distribution of z.
    """
    hist = torch.histc(z, bins=bins, min=range[0], max=range[1])
    dist = hist / torch.sum(hist)  # Normalize to get a probability distribution
    return dist

def estimate_tv_distance_hist(samples1, samples2, bins=50):
    """
    Estimate Total Variation distance between two distributions using histograms.
    
    Args:
        samples1: Array of shape (n_samples, n_dims) from p1
        samples2: Array of shape (n_samples, n_dims) from p2
        bins: Number of bins per dimension (or a list for different bin counts per dimension)
    
    Returns:
        float: Estimated TV distance
    """
    # Ensure samples are numpy arrays
    samples1 = np.asarray(samples1)
    samples2 = np.asarray(samples2)
    
    # Check dimensions
    assert samples1.shape[1] == samples2.shape[1], "Sample dimensions must match"
    n_dims = samples1.shape[1]
    
    # Compute histograms (normalized to form probability distributions)
    if n_dims == 1:
        hist1, edges = np.histogram(samples1, bins=bins, density=True)
        hist2, _ = np.histogram(samples2, bins=edges, density=True)
    else:
        # For multidimensional data, use np.histogramdd
        hist_range = [[min(np.min(samples1[:, i]), np.min(samples2[:, i])), 
                       max(np.max(samples1[:, i]), np.max(samples2[:, i]))] 
                      for i in range(n_dims)]
        hist1, edges = np.histogramdd(samples1, bins=bins, range=hist_range, density=True)
        hist2, _ = np.histogramdd(samples2, bins=bins, range=hist_range, density=True)
    
    # Compute TV distance
    tv_distance = 0.5 * np.sum(np.abs(hist1 - hist2))
    
    return tv_distance

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
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='base', train_epoch=0, activation=None, target_seq_len=None, drop_out=None, from_scratch=False):
        super().__init__()
        self.classification = (not isinstance(output_shape, tuple)) and (output_shape != 1)
        self.output_raw = True
        print("cur_classification: ", self.classification)
        
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
        set_grad_state(self.predictor, False)

        if use_embedder:
            self.embedder = Embeddings2D(input_shape, patch_size=patch_size, config=self.model.config, embed_dim=embed_dim, img_size=img_size)
            embedder_init(self.model.swin.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)
            self.model.swin.embeddings = self.embedder  


    def forward(self, x):
        if self.output_raw:
            return self.model.swin.embeddings(x)[0]
            
        x = self.model(x).logits

        return self.predictor(x)


class wrapper1D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='roberta', train_epoch=0, activation=None, target_seq_len=512, drop_out=None, from_scratch=False):
        super().__init__()

        self.dense = False
        self.output_raw = True
        self.weight = weight
        self.output_shape = output_shape

        if isinstance(output_shape, tuple):
            self.dense = True

        
        
        modelname = 'roberta-base' 
        configuration = AutoConfig.from_pretrained(modelname)
        if drop_out is not None:
            configuration.hidden_dropout_prob = drop_out
            configuration.attention_probs_dropout_prob = drop_out
        self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)

        if use_embedder:
            self.embedder = Embeddings1D(input_shape, config=self.model.config, embed_dim= 768, target_seq_len= target_seq_len, dense=self.dense)
            embedder_init(self.model.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)    
        else:
            self.embedder = nn.Identity()

        
        self.model.embeddings = embedder_placeholder()
        if self.dense:
            self.model.pooler = nn.Identity()
            self.predictor = adaptive_pooler(out_channel = output_shape[-2] * self.embedder.stack_num, output_shape=output_shape, dense=True)
        else:
            self.model.pooler = adaptive_pooler()
            self.predictor = nn.Linear(in_features=768, out_features=output_shape)   
        

        if activation == 'sigmoid':
            self.predictor = nn.Sequential(self.predictor, nn.Sigmoid())  
            
        set_grad_state(self.model, False)
        set_grad_state(self.predictor, False)


    def forward(self, x):
        
        if self.output_raw:
            return self.embedder(x) 

        x = self.embedder(x)

        if self.dense:
            x = self.model(inputs_embeds=x)['last_hidden_state']
            x = self.predictor(x)
        else:
            x = self.model(inputs_embeds=x)['pooler_output']
            x = self.predictor(x)

        if x.shape[1] == 1 and len(x.shape) == 2:
            x = x.squeeze(1)

        return x

#######################################################################################################
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        cached = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
        print(f"Allocated GPU memory: {allocated:.2f} MB")
        print(f"Cached GPU memory: {cached:.2f} MB")
    else:
        print("CUDA is not available.")
#######################################################################################################
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


#########################################################################################################################################################################
def get_pretrain_model2D_feature(args,root,sample_shape, num_classes, source_classes = 10):
    ###################################### train predictor 
    """
    get train model and feature: 
       + get trained predictor but embedder is convolution 
    """
    print("get src_model and src_feature...")
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    IMG_SIZE = 224 if args.weight == 'tiny' or args.weight == 'base' else 196
    num_classes = 10
    print("num class: ", num_classes)    
    src_model = wrapper2D(sample_shape, num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out)
    src_model = src_model.to(args.device)
    src_model.output_raw = False
    optimizer = optim.AdamW(
         src_model.parameters(),
         lr=args.lr if hasattr(args, 'lr') else 1e-4,
         weight_decay=0.05
     )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
         optimizer,
         T_max=args.embedder_epochs/10
     )
    criterion = nn.CrossEntropyLoss(
         label_smoothing=0.1 if hasattr(args, 'label_smoothing') else 0.0  # Optional smoothing
     )
    src_model.train()
    set_grad_state(src_model.predictor, True)
    print("trainabel params count :  ",count_trainable_params(src_model))
    print("trainable params: ")  
    for name, param in src_model.named_parameters():
       if param.requires_grad:
          print(name)
     #print("model architechture: ", src_model)      
    for epoch in range(args.embedder_epochs//10):
        running_loss = 0.0 
        correct = 0  
        total = 0
        for i, data in enumerate(src_train_loader):
             x_, y_ = data 
             x_ = x_.to(args.device)
             y_ = y_.to(args.device)
             x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
             optimizer.zero_grad()
             out = src_model(x_)
             loss = criterion(out, y_)
             loss.backward()
             optimizer.step()
             running_loss += loss.item()
             _, predicted = torch.max(out, 1)  # Get the index of max log-probability
             total += y_.size(0)
             correct += (predicted == y_).sum().item()
         
        scheduler.step()
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{args.embedder_epochs//10}], '
               f'Average Loss: {running_loss/len(src_train_loader):.4f}'
               f' Accuracy: {accuracy:.2f}%')  
         
    ##### set output_raw 
    src_model.output_raw = True
    src_model.eval()
    ##### get source feature from cifar10
    src_feats = []
    src_ys = []
    for i, data in enumerate(src_train_loader):
            x_, y_ = data 
            x_ = x_.to(args.device)
            x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
            out = src_model(x_)
            if len(out.shape) > 2:
                out = out.mean(1)

            src_ys.append(y_.detach().cpu())
            src_feats.append(out.detach().cpu())
            
    src_feats = torch.cat(src_feats, 0)
    src_ys = torch.cat(src_ys, 0).long()
    src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
    ##### clearn cache
    del src_ys, src_feats, src_train_loader
    torch.cuda.empty_cache()
    ###### get pre-trained pridictor
    #src_model.predictor = get_top_k_predictor_2Dmodel(source_classes)
    ##### return src_model and src_train_dataset         
    return src_model, src_train_dataset
    
###############################################################################################################################################
def get_top_k_predictor_2Dmodel(k=1000):
    # Load pre-trained Swin model (ImageNet-22K)
    arch_name = "microsoft/swin-base-patch4-window7-224-in22k"
    output_dim = 1024  # Swin Base hidden size
    model = SwinForImageClassification.from_pretrained(arch_name)

    # Extract the original classifier (21K classes)
    old_classifier = model.classifier  # Linear(1024, 21843)
    
    # Copy weight from the top K classes
    old_weights = old_classifier.weight.data  # Shape: (21843, 1024)
    new_weights = old_weights[:k, :].clone()  # Shape: (k, 1024), use `.clone()` to avoid modifying the original tensor

    # Copy bias if it exists
    if old_classifier.bias is not None:
        new_bias = old_classifier.bias.data[:k].clone()  # Shape: (k,)
    else:
        new_bias = None

    # Create a new classifier with K classes
    new_classifier = nn.Linear(output_dim, k)

    # Copy weights & bias safely
    with torch.no_grad():
        new_classifier.weight.copy_(new_weights)
        if new_bias is not None:
            new_classifier.bias.copy_(new_bias)

    # Free memory properly
    del model  
    torch.cuda.empty_cache()  # Optional: Clear GPU memory if needed

    return new_classifier
        

##############################################################################################################################################
def feature_matching_tgt_model(args,root , tgt_model, src_train_dataset):
    """
    Embedder training using optimal transport to minimize source and target feture distribution: 
      + Early implement using OTDD
      + Feature work using Total Variance distance 
      + return tgt_model
    """ 
    
    print("feature matching....")
    ##### check tgt_model
    set_grad_state(tgt_model.embedder, True)
    print("trainabel params count :  ",count_trainable_params(tgt_model))
    print("trainable params: ")  
    for name, param in tgt_model.named_parameters():
      if param.requires_grad:
         print(name)
    ### load tgt_loader
    print("load tgt dataset...") 
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
    #### fix fowrad flow and set trainable to tgt_model
    tgt_model.output_raw = True
    tgt_model = tgt_model.to(args.device).train()
    
    #### init score function 
    score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)
    ##### Test for new test function
    # src_feature =  src_train_dataset.tensors[0].numpy()
    # new_score_func = partial(OptimalTV, sample1 = src_feature)
    ##### get optimizer
    args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder')
    tgt_model_optimizer.zero_grad()
    ##### train embedder with score_func
     
    total_losses, times, embedder_stats = [], [], []
    print("begin feature matching...")
    for ep in range(args.embedder_epochs):   

        total_loss = 0    
        time_start = default_timer()
        feats = []
        datanum = 0
        ##### shuffle dataset 
        
        shuffled_loader = torch.utils.data.DataLoader(
                tgt_train_loader.dataset,
                batch_size=tgt_train_loader.batch_size,
                shuffle=True,  # Enable shuffling to permute the order
                num_workers=tgt_train_loader.num_workers,
                pin_memory=tgt_train_loader.pin_memory)
        ##### begin training
             
        for j, data in enumerate(shuffled_loader):
            
            if transform is not None:
                x, y, z = data
            else:
                x, y = data 
                
            x = x.to(args.device)
            out = tgt_model(x)
            
            feats.append(out)
            datanum += x.shape[0]
                
            if datanum > args.maxsamples:
                  break

        feats = torch.cat(feats, 0).mean(1)
        if feats.shape[0] > 1:
            #feats_np = feats.numpy()
            loss = (len(feats)/len(tgt_train_loader))*score_func(feats)
            loss.backward()
            total_loss += loss.item()

        time_end = default_timer()  
        times.append(time_end - time_start) 

        total_losses.append(total_loss)
        embedder_stats.append([total_losses[-1], times[-1]])
        print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\totdd loss:", "%.4f" % total_losses[-1])

        tgt_model_optimizer.step()
        tgt_model_scheduler.step()
        tgt_model_optimizer.zero_grad()

    del tgt_train_loader 
    torch.cuda.empty_cache()

    tgt_model.output_raw = False

    return tgt_model
###############################################################################################################################################    
def label_matching_src_2Dmodel(args,root, src_model, tgt_embedder, num_classes, src_num_classes):
    """
    Label matching by minimize -H(Y_t| Y_s): 
      + Generate dummy label for target data.
      + Compute emperical P(Y_t| Y_s).
      + Minimize -H(Y_t| Y_s)
      + Return src_model without predictor
    """  
    print("label matching with src model...")
    ##### check src_model
    src_model.embedder = tgt_embedder
    src_model.model.swin.embeddings = src_model.embedder
    set_grad_state(src_model.model, True)
    set_grad_state(src_model.embedder, False) #86753474
    print("trainabel params count :  ",count_trainable_params(src_model))
    print("trainable params: ")
    src_model.output_raw = False
    src_model = src_model.to(args.device).train()  
    # for name, param in src_model.named_parameters():
    #   if param.requires_grad:
    #      print(name)
         
    ##### load tgt dataset
    print("load tgt dataset...")
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
    print("infer label...")
    if args.infer_label:
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader)
        
    else: 
        num_classes_new = num_classes
    
    ####### get optimizer
    args, src_model, optimizer, scheduler = get_optimizer_scheduler(args, src_model, module=None, n_train=n_train)
    optimizer.zero_grad()         
    ####### train with dummy label 
    print("Training with dummy label...")
    ###### config for testing
    label_matching_ep = (args.epochs//10) + 1 
    max_sample =  args.label_maxsamples
    total_losses, times, stats = [], [], []
    ###### begin training with dummy label
    shuffled_loader = torch.utils.data.DataLoader(
                tgt_train_loader.dataset,
                batch_size=tgt_train_loader.batch_size,
                shuffle=True,  # Enable shuffling to permute the order
                num_workers=tgt_train_loader.num_workers,
                pin_memory=tgt_train_loader.pin_memory)
    
    for ep in range(label_matching_ep):
        total_loss = 0    
        time_start = default_timer()    
        dummy_label = []
        dummy_probability = []
        target_label = []   
        datanum = 0 
        for j, data in enumerate(shuffled_loader):
                
               if transform is not None:
                  x, y, z = data
               else:
                  x, y = data 
                
               x = x.to(args.device)
               y = y.to(args.device)
               out = src_model(x)
               out = F.softmax(out, dim=-1)
               probability, predicted = torch.max(out, 1) 
               dummy_label.append(predicted)
               dummy_probability.append(probability)
               target_label.append(y)
               datanum += x.shape[0] 
               #print("datanum: ", datanum)
               #get_gpu_memory_usage()
               if datanum >= max_sample:
                   #print("datanum when backward: ", datanum)
                   #get_gpu_memory_usage()
                   dummy_labels_tensor = torch.cat(dummy_label, dim=0)
                   dummy_probs_tensor = torch.cat(dummy_probability, dim=0)  # This is a tensor
                   target_label_tensor = torch.cat(target_label, dim=0)
                   loss = (datanum/len(shuffled_loader))*weighted_CE_loss(dummy_labels_tensor,dummy_probs_tensor ,target_label_tensor,src_num_classes,num_classes_new)
                   loss.backward()
                   total_loss += loss.item()
                   optimizer.step()
                   optimizer.zero_grad()
                   dummy_label = []
                   target_label = []
                   dummy_probability = []
                   datanum = 0
        ###################### handle leftover dataset.
        if (datanum >= max_sample//2):             
            dummy_labels_tensor = torch.cat(dummy_label, dim=0)
            dummy_probs_tensor = torch.cat(dummy_probability, dim=0)  # This is a tensor
            target_label_tensor = torch.cat(target_label, dim=0)
            loss = (datanum/len(shuffled_loader))*weighted_CE_loss(dummy_labels_tensor,dummy_probs_tensor ,target_label_tensor,10,num_classes_new)
            loss.backward()
            total_loss += loss.item()
        ##############################
        time_end = default_timer()  
        times.append(time_end - time_start) 

        total_losses.append(total_loss)
        stats.append([total_losses[-1], times[-1]])
        print("[label matching ", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\tCE loss:", "%.4f" % total_losses[-1])

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()           

            
    return src_model  
####################################################################################################################################################################################
def label_matching_src_1Dmodel(args,root, src_model, tgt_embedder,num_classes ,src_num_classes):
    """
    Label matching by minimize -H(Y_t| Y_s): 
      + Generate dummy label for target data.
      + Compute emperical P(Y_t| Y_s).
      + Minimize -H(Y_t| Y_s)
      + Return src_model without predictor
    """  
    print("label matching with src model...")
    ##### check src_model
    src_model.embedder = tgt_embedder
    
    set_grad_state(src_model.model, True)
    set_grad_state(src_model.embedder, False) #86753474
    print("trainabel params count :  ",count_trainable_params(src_model))
    print("trainable params: ")
    src_model.output_raw = False
    src_model = src_model.to(args.device).train()  
    for name, param in src_model.named_parameters():
      if param.requires_grad:
         print(name)
         
    ##### load tgt dataset
    print(src_model)
    print("load tgt dataset...")
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
    print("infer label...")
    if args.infer_label:
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader)
        
    else: 
        num_classes_new = num_classes
    
    ####### get optimizer
    args, src_model, optimizer, scheduler = get_optimizer_scheduler(args, src_model, module=None, n_train=n_train)
    optimizer.zero_grad()         
    ####### train with dummy label 
    print("Training with dummy label...")
    ###### config for testing
    label_matching_ep = (args.epochs//10) + 1 
    max_sample = args.label_maxsamples
    total_losses, times, stats = [], [], []
    ###### begin training with dummy label
    shuffled_loader = torch.utils.data.DataLoader(
                tgt_train_loader.dataset,
                batch_size=tgt_train_loader.batch_size,
                shuffle=True,  # Enable shuffling to permute the order
                num_workers=tgt_train_loader.num_workers,
                pin_memory=tgt_train_loader.pin_memory)
    
    for ep in range(label_matching_ep):
        total_loss = 0    
        time_start = default_timer()    
        dummy_label = []
        dummy_probability = []
        target_label = []   
        datanum = 0 
        for j, data in enumerate(shuffled_loader):
                
               if transform is not None:
                  x, y, z = data
               else:
                  x, y = data 
                
               x = x.to(args.device)
               y = y.to(args.device)
               out = src_model(x)
               out = F.softmax(out, dim=-1)
               probability, predicted = torch.max(out, 1) 
               dummy_label.append(predicted)
               dummy_probability.append(probability)
               target_label.append(y)
               datanum += x.shape[0]
            #    print("datanum: ", datanum)
            #    get_gpu_memory_usage() 
               if datanum >= max_sample:
                #    print("run backward: ")
                #    get_gpu_memory_usage()
                   
                   dummy_labels_tensor = torch.cat(dummy_label, dim=0)
                   dummy_probs_tensor = torch.cat(dummy_probability, dim=0)  # This is a tensor
                   target_label_tensor = torch.cat(target_label, dim=0)
                   loss = (datanum/len(shuffled_loader))*weighted_CE_loss(dummy_labels_tensor,dummy_probs_tensor ,target_label_tensor,src_num_classes,num_classes_new)
                   loss.backward()
                   optimizer.step()
                   optimizer.zero_grad()
                   total_loss += loss.item()
                   dummy_label = []
                   target_label = []
                   dummy_probability = []
                   datanum = 0
                   #print("after grad")
                   #get_gpu_memory_usage()
        ###################### handle leftover dataset. 
        if (datanum >= max_sample//2):             
            dummy_labels_tensor = torch.cat(dummy_label, dim=0)
            dummy_probs_tensor = torch.cat(dummy_probability, dim=0)  # This is a tensor
            target_label_tensor = torch.cat(target_label, dim=0)
            loss = (datanum/len(shuffled_loader))*weighted_CE_loss(dummy_labels_tensor,dummy_probs_tensor ,target_label_tensor,10,num_classes_new)
            loss.backward()
            total_loss += loss.item()
        ##############################
        time_end = default_timer()  
        times.append(time_end - time_start) 

        total_losses.append(total_loss)
        stats.append([total_losses[-1], times[-1]])
        print("[label matching ", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\tCE loss:", "%.4f" % total_losses[-1])

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()           

            
    return src_model  
####################################################################################################################################################################################
def CE_loss(dummy_labels,dummy_probs , target_label, src_num_classes, tgt_num_classes):
    """ Compute negative conditional entropy between target label and source label -H(Y_t| Y_s).

    Args:
        dummy_label (_type_).
        dummy_probability (_type_).
        target_label (_type_): list of target label.
        src_num_classes (_type_): number of src classes.
        tgt_num_classes (_type_): number of tgt classes.

    Returns:
        tensor : -H(Y_t| Y_s)
    """
    
    device = dummy_probs.device
    dummy_labels = dummy_labels.to(device).detach()  # No gradients for indices
    target_label = target_label.to(device).detach()
    dummy_probs = dummy_probs.to(device)
    
    ##### get observed classes 
    # Get unique observed classes in this batch
    observed_src_classes = torch.unique(dummy_labels)  # e.g., [0, 1]
    observed_tgt_classes = torch.unique(target_label)
    
    P_temp = torch.zeros(src_num_classes, tgt_num_classes, device=device, dtype=torch.float32)
    
    # Manually accumulate dummy_probs into P_temp
    batch_size = dummy_labels.size(0)
    for i in range(batch_size):
        src_idx = dummy_labels[i]
        tgt_idx = target_label[i]
        P_temp[src_idx, tgt_idx] += dummy_probs[i]  # In-place on P_temp (no grad)

    # Create P_full with gradients, using P_temp as the initial value
    P_full = P_temp.clone().requires_grad_(True)
    # Filter P to observed classes only
    P = P_full[observed_src_classes][:, observed_tgt_classes]
    # Compute marginal probability p(dummy_label) over observed target classes
    marginal_p = P.sum(dim=1)  # Shape: [num_observed_src], e.g., [2]

    # Avoid division by zero
    valid_src_mask = marginal_p > 0  # Shape: [num_observed_src]
    marginal_p_safe = marginal_p[valid_src_mask]  # Shape: [num_valid_src]

    if marginal_p_safe.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=dummy_probs.requires_grad)

    # Filter P to valid source classes
    P_filtered = P[valid_src_mask]  # Shape: [num_valid_src, num_observed_tgt]

    # Compute conditional probability p(target_label | dummy_label)
    P_cond = P_filtered / marginal_p_safe.unsqueeze(1)  # Shape: [num_valid_src, num_observed_tgt]

    # Mask to avoid log(0)
    mask = P_filtered > 0  # Shape: [num_valid_src, num_observed_tgt]
    P_cond_safe = P_cond[mask]  # Shape: [num_valid_elements]

    if P_cond_safe.numel() > 0:
        loss = -torch.log(P_cond_safe).sum()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=dummy_probs.requires_grad)

    return loss  
#########################################################################################################################################################################
def weighted_CE_loss(dummy_labels, dummy_probs, target_label, src_num_classes, tgt_num_classes):
    """ Compute weighted negative conditional entropy -H(Y_t | Y_s) over observed target classes.

    Args:
        dummy_labels: Tensor [batch_size], predicted source labels (e.g., [0, 1, 0])
        dummy_probs: Tensor [batch_size], probabilities for dummy_labels (e.g., [0.9, 0.8, 0.7])
        target_label: Tensor [batch_size], true target labels (e.g., [0, 1, 0])
        src_num_classes: int, number of source classes (e.g., 3)
        tgt_num_classes: int, number of target classes (e.g., 2 or more)

    Returns:
        torch.Tensor: Weighted -H(Y_t | Y_s)
    """
    # Handle case where dummy_probs is a list (safety check)
    if isinstance(dummy_probs, list):
        dummy_probs = torch.cat(dummy_probs, dim=0)

    # Ensure tensors are on the same device and detach labels
    device = dummy_probs.device
    dummy_labels = dummy_labels.to(device).detach()  # Shape: [batch_size], e.g., [3]
    target_label = target_label.to(device).detach()  # Shape: [batch_size], e.g., [3]
    dummy_probs = dummy_probs.to(device)  # Shape: [batch_size], e.g., [3]

    # Get unique observed classes in this batch
    observed_src_classes = torch.unique(dummy_labels)  # e.g., [0, 1]
    observed_tgt_classes = torch.unique(target_label)  # e.g., [0, 1]
    
    # Initialize temporary accumulation tensor without gradients
    P_temp = torch.zeros(src_num_classes, tgt_num_classes, device=device, dtype=torch.float32)
    
    # Manually accumulate dummy_probs into P_temp
    batch_size = dummy_labels.size(0)
    for i in range(batch_size):
        src_idx = dummy_labels[i]  # e.g., 0, 1, 0
        tgt_idx = target_label[i]  # e.g., 0, 1, 0
        P_temp[src_idx, tgt_idx] += dummy_probs[i]  # Add prob to P[src_idx][tgt_idx]

    # Create P_full with gradients
    P_full = P_temp.clone().requires_grad_(True)  # Shape: [3, tgt_num_classes], e.g., [3, 2]

    # Filter P to observed classes
    P = P_full[observed_src_classes][:, observed_tgt_classes]  # Shape: [num_observed_src, num_observed_tgt], e.g., [2, 2]

    # Compute marginal probability p(dummy_label) over observed target classes
    marginal_p = P.sum(dim=1)  # Shape: [num_observed_src], e.g., [2]

    # Avoid division by zero
    valid_src_mask = marginal_p > 0
    marginal_p_safe = marginal_p[valid_src_mask]  # Shape: [num_valid_src]

    if marginal_p_safe.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=dummy_probs.requires_grad)

    # Filter P to valid source classes
    P_filtered = P[valid_src_mask]  # Shape: [num_valid_src, num_observed_tgt], e.g., [2, 2]

    # Compute conditional probability p(target_label | dummy_label)
    P_cond = P_filtered / marginal_p_safe.unsqueeze(1)  # Shape: [num_valid_src, num_observed_tgt], e.g., [2, 2]

    # Compute loss: sum(-log(p(target_label | Y_s))) weighted by number of samples per target class
    loss = 0.0
    for tgt_class in observed_tgt_classes:
        # Number of samples with this target class
        num_samples = (target_label == tgt_class).sum().item()  # e.g., 2 for tgt=0, 1 for tgt=1
        
        # Indices in observed_tgt_classes where this target class appears
        tgt_idx = (observed_tgt_classes == tgt_class).nonzero(as_tuple=True)[0]  # Index of tgt_class in observed_tgt_classes
        
        # Conditional probabilities for this target class
        P_cond_tgt = P_cond[:, tgt_idx]  # Shape: [num_valid_src, 1], e.g., [2, 1]
        
        # Mask to avoid log(0)
        mask = P_filtered[:, tgt_idx] > 0  # Shape: [num_valid_src, 1]
        P_cond_safe = P_cond_tgt[mask]  # Shape: [num_valid_elements]
        
        if P_cond_safe.numel() > 0:
            # Weighted contribution: num_samples * sum(-log(p(target_label=tgt_class | Y_s)))
            loss += num_samples * (-torch.log(P_cond_safe).sum())

    if loss == 0.0:  # If no valid probabilities were found
        return torch.tensor(0.0, device=device, requires_grad=dummy_probs.requires_grad)

    return loss
#########################################################################################################################################################################
def get_src_train_dataset_1Dmodel(args,root):
    """
    get source train dataset with backbone Roberta: 
       + return source train dataset.
    """
    
    ####### load src train dataset.
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
    zero_labels = torch.zeros_like(src_ys)
    src_train_dataset = torch.utils.data.TensorDataset(src_feats, zero_labels)
    return src_train_dataset

#########################################################################################################################################################################
def get_tgt_model(args, root, sample_shape, num_classes, loss, add_loss=False, use_determined=False, context=None, opid=0):
    
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    if len(sample_shape) == 4:
        IMG_SIZE = 224 if args.weight == 'tiny' or args.weight == 'base' else 196
            
        src_model = wrapper2D(sample_shape, num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out)
        src_model = src_model.to(args.device).eval()
            
        src_feats = []
        src_ys = []
        for i, data in enumerate(src_train_loader):
            x_, y_ = data 
            x_ = x_.to(args.device)
            x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
            out = src_model(x_)
            if len(out.shape) > 2:
                out = out.mean(1)

            src_ys.append(y_.detach().cpu())
            src_feats.append(out.detach().cpu())
        src_feats = torch.cat(src_feats, 0)
        src_ys = torch.cat(src_ys, 0).long()
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)        
        del src_model    

    else:
        src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
        
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
        
    if args.infer_label:
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader)
    else:
        num_classes_new = num_classes

    print("src feat shape", src_feats.shape, src_ys.shape, "num classes", num_classes_new) 

    tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, num_classes_new)

    wrapper_func = wrapper1D if len(sample_shape) == 3 else wrapper2D
    tgt_model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
    tgt_model = tgt_model.to(args.device).train()

    args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder')
    tgt_model_optimizer.zero_grad()

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
    
    for ep in range(args.embedder_epochs):   

        total_loss = 0    
        time_start = default_timer()

        for i in np.random.permutation(num_classes_new):
            feats = []
            datanum = 0

            for j, data in enumerate(tgt_train_loaders[i]):
                
                if transform is not None:
                    x, y, z = data
                else:
                    x, y = data 
                
                x = x.to(args.device)
                out = tgt_model(x)
                feats.append(out)
                datanum += x.shape[0]
                
                if datanum > args.maxsamples: break

            feats = torch.cat(feats, 0).mean(1)
            if feats.shape[0] > 1:
                loss = tgt_class_weights[i] * score_func(feats)
                loss.backward()
                total_loss += loss.item()

        time_end = default_timer()  
        times.append(time_end - time_start) 

        total_losses.append(total_loss)
        embedder_stats.append([total_losses[-1], times[-1]])
        print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\totdd loss:", "%.4f" % total_losses[-1])

        tgt_model_optimizer.step()
        tgt_model_scheduler.step()
        tgt_model_optimizer.zero_grad()

    del tgt_train_loader, tgt_train_loaders
    torch.cuda.empty_cache()

    tgt_model.output_raw = False

    return tgt_model, embedder_stats


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


def load_by_class(loader, num_classes):
    train_set = loader.dataset
    subsets = {}

    if len(train_set.__getitem__(0)) == 3:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    else:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
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