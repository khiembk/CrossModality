import embedder
from utils import count_params, count_trainable_params, calculate_stats
import tqdm
from embedder import wrapper2D
import numpy as np
import random
import torch
import os 
import argparse
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling, RobertaForTokenClassification
import torch.nn as nn
from lp_embedder import Embeddings2D
from timeit import default_timer
from functools import partial
from SVFT import LinearWithSVFT, create_and_replace_modules , get_target_modules_list
from utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, l2, MMD_loss
from task_configs import set_decoder_trainable
from task_configs import get_data, get_config, get_metric, get_optimizer_scheduler, set_trainable
from Contrastive_Embedder import get_SVFT_model
from main import load_state
from types import SimpleNamespace

class wrapper2D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='base', train_epoch=0, activation=None, target_seq_len=None, drop_out=None, from_scratch=False, warm_init = True):
        super().__init__()
        self.classification = (not isinstance(output_shape, tuple)) and (output_shape != 1)
        self.output_raw = True

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

        # set_grad_state(self.model, False)
        # set_grad_state(self.predictor, False)

        if use_embedder:
            self.embedder = Embeddings2D(input_shape, patch_size=patch_size, config=self.model.config, embed_dim=embed_dim, img_size=img_size)
            embedder_init(self.model.swin.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)
            self.model.swin.embeddings = self.embedder  

    def set_bodymodel_trainble(self):
        set_grad_state(self.model, True)
        set_grad_state(self.predictor, True)
        
    def forward(self, x):
        if self.output_raw:
            return self.model.swin.embeddings(x)[0]
            
        x = self.model(x).logits

        return self.predictor(x)

class wrapper2D_pretrain(torch.nn.Module): 
    def __init__(self, input_shape, output_shape, lora_rank=1, use_embedder=True, weight='base', train_epoch=0, activation=None, target_seq_len=None, drop_out=None, from_scratch=False, rankLoRA=1, warm_init=True, classification=None, train_embedder=False):
        super().__init__()
        self.classification = (not isinstance(output_shape, tuple)) and (output_shape != 1)
        self.output_raw = True
        self.train_predictor = False
        self.train_embedder = train_embedder
        if classification is not None:
            self.classification = classification
        
        # Model configuration
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

        modelclass = SwinForImageClassification if self.classification else SwinForMaskedImageModeling
        self.model = modelclass.from_pretrained(arch_name)
        self.model.config.image_size = img_size
        if drop_out is not None:
            self.model.config.hidden_dropout_prob = drop_out
            self.model.config.attention_probs_dropout_prob = drop_out
        self.model = modelclass.from_pretrained(arch_name, config=self.model.config) if not from_scratch else modelclass(self.model.config)

        # Setup classifier or predictor
        if self.classification:
            self.model.pooler = nn.AdaptiveAvgPool1d(1)
            self.model.classifier = nn.Identity()
            self.predictor = nn.Linear(in_features=output_dim, out_features=output_shape)
        else:
            self.pool_seq_dim = adaptive_pooler(output_shape[1] if isinstance(output_shape, tuple) else 1)
            self.pool = nn.AdaptiveAvgPool2d(input_shape[-2:])
            self.predictor = nn.Sequential(self.pool_seq_dim, self.pool)

        # Inject LoRA only into the last transformer block
        #self.model = self.get_rank_all_layer(self.model)
        

        # Embedding layer setup
        if use_embedder:
            self.embedder = Embeddings2D(input_shape, patch_size=patch_size, config=self.model.config, embed_dim=embed_dim, img_size=img_size)
            #set_grad_state(self.embedder, True)
            self.model.swin.embeddings = self.embedder  

    def get_rank_all_layer(self, model):
        
        transformer_blocks = model.swin.encoder.layers
        for layer in transformer_blocks:
            for block in layer.blocks:
                for sub_layer_name, sub_layer in block.named_children():
                    
                    if sub_layer_name in ["attention","output", "intermediate"] : 
                        if sub_layer_name == "attention":
                            print("Attention shape: ",  sub_layer.self.query.weight.shape)
                            cur_shape = sub_layer.self.query.weight.shape
                            cur_weight = sub_layer.self.query.weight.detach().cpu().numpy()
                            rank_query = np.linalg.matrix_rank(cur_weight)
                            rank_key = np.linalg.matrix_rank(sub_layer.self.key.weight.detach().cpu().numpy())
                            print("rank of query: ", rank_query)
                            print("rank of key: ", rank_key)
                            rank_value = np.linalg.matrix_rank(sub_layer.self.value.weight.detach().cpu().numpy())
                            print("rank of value: ", rank_value)
                        if sub_layer_name == "intermediate":
                            print("Intermediate shape: ", sub_layer.dense.weight.shape)
                            cur_shape = sub_layer.dense.weight.shape
                            cur_weight = sub_layer.dense.weight.detach().cpu().numpy()
                            rank = np.linalg.matrix_rank(cur_weight)
                            print("rank of Intermediate: ", rank)
                        if sub_layer_name == "output":
                            print("Output shape: ",sub_layer.dense.weight.shape)   
                            cur_shape =  sub_layer.dense.weight.shape
                            cur_weight = sub_layer.dense.weight.detach().cpu().numpy()
                            rank = np.linalg.matrix_rank(cur_weight)
                            print("rank of Output: ", rank)

            downsample = layer.downsample
            if (downsample is not None):
                print("downsample shape: ", downsample.reduction.weight.shape)
                cur_shape =  downsample.reduction.weight.shape
                cur_weight = downsample.reduction.weight.detach().cpu().numpy()
                rank = np.linalg.matrix_rank(cur_weight)
                print("rank of downsample: ", rank)
        return model      

    def get_differnt_rank_all_layer_with_model(self, model, model1):
        model_weight = []
        model1_weight = []
        transformer_blocks = model.model.swin.encoder.layers

        for layer in transformer_blocks:
            for block in layer.blocks:
                for sub_layer_name, sub_layer in block.named_children():
                    
                    if sub_layer_name in ["attention","output", "intermediate"] : 
                        if sub_layer_name == "attention":
                            cur_weight_query = sub_layer.self.query.weight.detach().cpu().numpy()
                            cur_weight_key = sub_layer.self.key.weight.detach().cpu().numpy()
                            cur_weight_value = sub_layer.self.value.weight.detach().cpu().numpy()
                            model_weight.append(cur_weight_query)
                            model_weight.append(cur_weight_key)
                            model_weight.append(cur_weight_value)
                        if sub_layer_name == "intermediate":
                            cur_weight = sub_layer.dense.weight.detach().cpu().numpy()
                            model_weight.append(cur_weight)
                        if sub_layer_name == "output":   
                            cur_weight = sub_layer.dense.weight.detach().cpu().numpy()
                            model_weight.append(cur_weight)

            downsample = layer.downsample
            if (downsample is not None):
                cur_weight = downsample.reduction.weight.detach().cpu().numpy()
                model_weight.append(cur_weight)

        transformer1_blocks = model1.model.swin.encoder.layers

        for layer in transformer1_blocks:
            for block in layer.blocks:
                for sub_layer_name, sub_layer in block.named_children():
                     if sub_layer_name in ["attention","output", "intermediate"] : 
                        if sub_layer_name == "attention":
                            cur_weight_query = sub_layer.self.query.weight.detach().cpu().numpy()
                            cur_weight_key = sub_layer.self.key.weight.detach().cpu().numpy()
                            cur_weight_value = sub_layer.self.value.weight.detach().cpu().numpy()
                            model1_weight.append(cur_weight_query)
                            model1_weight.append(cur_weight_key)
                            model1_weight.append(cur_weight_value)
                        if sub_layer_name == "intermediate":
                            cur_weight = sub_layer.dense.weight.detach().cpu().numpy()
                            model1_weight.append(cur_weight)
                        if sub_layer_name == "output":   
                            cur_weight = sub_layer.dense.weight.detach().cpu().numpy()
                            model1_weight.append(cur_weight)

            downsample = layer.downsample
            if (downsample is not None):
                cur_weight = downsample.reduction.weight.detach().cpu().numpy()
                model1_weight.append(cur_weight)

        for index in range(len(model_weight)):
            cur = model_weight[index] - model1_weight[index]
            rank = np.linalg.matrix_rank(cur)
            print("rank of lora: ",rank)        

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


def test1D_model():
    wrapper_funcLORA = embedder.wrapper1DLORA 
    sample_shape =  (1, 1, 1000)
    output_shape = 3
    model = wrapper_funcLORA(sample_shape,output_shape)   
    #print(model)
    print("trainable params: ", count_trainable_params(model))
    print("all params: ", count_params(model))
    for name, param in model.named_parameters():
       if not param.requires_grad:
           print(f"Layer: {name}")

def test2D_model():
    sample_shape = (3,224,224)
    output_shape = 1
    model = wrapper2D(sample_shape,output_shape)
    
    # for param in model.parameters():
    #         param.requires_grad = False
    

    lora_target_modules = ["query", "value", "key", "projection","dense" ]
    print(f"Target Modules: {lora_target_modules}")
    off_diag = 1
    assign_svft_layer = partial(LinearWithSVFT, 
                                    off_diag=off_diag, 
                                    pattern= "banded", 
                                    rank= None, 
                                    fill_orthonormal= False)
        
    create_and_replace_modules(model, get_target_modules_list(model, lora_target_modules), assign_svft_layer)
    #set_decoder_trainable(model)
    print("trainable params: ", count_trainable_params(model))
    print("all params: ", count_params(model))
    for name, param in model.named_parameters():
       if not param.requires_grad:
           print(f"Layer: {name}")
    
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def main():
    set_seed()
    # random_num = random.randint(1, 100)
    # print("random number: ", random_num)
    test2D_model()

def TestSVFT_scoring(use_determined ,args,info=None, context=None, lora_rank=1, mode = 'lora', save_per_ep = 1, DatasetRoot= None, log_folder = None, warm_init = True):
    set_seed()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #args.device = 'cuda' 
    print("The current device is: ", args.device)
    root = '/datasets' if use_determined else './datasets'
    if (DatasetRoot != None):
        root = DatasetRoot + '/datasets'

    print("Path folder dataset: ",root) 
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)
    dims, sample_shape, num_classes, loss, args = get_config(root, args)    
    model, embedder_stats = get_SVFT_model(args, root, sample_shape, num_classes, loss,lora_rank ,False, use_determined, context)
    
    print("all param count:", count_params(model))
    print("trainabel params count :  ",count_trainable_params(model))    
    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split)
    metric, compare_metrics = get_metric(root, args.dataset)
    decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
    model, ep_start, id_best, train_score, train_losses, embedder_stats_saved = load_state(use_determined, args, context, model, None, None, n_train, freq=args.validation_freq, test=True)
    embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    offset = 0 if ep_start == 0 else 1
    args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None if args.predictor_epochs == 0 or ep_start >= args.predictor_epochs else 'predictor', n_train=n_train)
    print("all param count:", count_params(model))
    print("trainabel params count :  ",count_trainable_params(model))  
    model = set_trainable(model)
    print("affter set trainable : ")
    print("all param count:", count_params(model))
    print("trainabel params count :  ",count_trainable_params(model))  
    #print learnabel 
    for name, param in model.named_parameters():
      if param.requires_grad:
         print(name)
         
    print("[check]trainabel params count :  ",count_trainable_params(model))      
    train_full = args.predictor_epochs == 0 or ep_start >= args.predictor_epochs
   
    if args.device == 'cuda':
        model.cuda()
        try:
            loss.cuda()
        except:
            pass
        if decoder is not None:
            decoder.cuda()

    print("\n------- Experiment Summary --------")
    print("id:", args.experiment_id)
    print("dataset:", args.dataset, "\tbatch size:", args.batch_size, "\tlr:", args.optimizer["params"]["lr"])
    print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)
    print("finetune method:", args.finetune_method)
    print("all param count:", count_params(model),)
    
    print("trainabel params count: %d  ",count_trainable_params(model))
    
    print("print model")
    print(model)
    model, ep_start, id_best, train_score, train_losses, embedder_statssaved = load_state(use_determined, args, context, model, optimizer, scheduler, n_train, freq=args.validation_freq)
    embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    train_time = []

    print("\n------- Start Training --------" if ep_start == 0 else "\n------- Resume Training --------")
    print("register hook")
    for name, params in  model.named_modules():
       if isinstance(params, LinearWithSVFT):
           params.register_gradient_hook()

    
    time_start = default_timer()
    train_loss = train_one_epoch(context, args, model, optimizer, scheduler, train_loader, loss, n_train, decoder, transform,mode =mode)
    train_time_ep = default_timer() -  time_start 
        
def train_one_epoch(context, args, model, optimizer, scheduler, loader, loss, temp, decoder=None, transform=None, mode = 'lora'):    

    model.train()             
    train_loss = 0
    optimizer.zero_grad()

    for i, data in enumerate(loader):

        if transform is not None:
            x, y, z = data
            z = z.to(args.device)
        else:
            x, y = data 
        
        x, y = x.to(args.device), y.to(args.device)
        out = model(x)

        if isinstance(out, dict):
            out = out['out']

        if decoder is not None:
            out = decoder.decode(out).view(x.shape[0], -1)
            y = decoder.decode(y).view(x.shape[0], -1)

        if transform is not None:
            out = transform(out, z)
            y = transform(y, z)

        if args.dataset[:4] == "DRUG":
            out = out.squeeze(1)
        
        l = loss(out, y)
        l.backward()
        train_loss += l.item()
        if i >= temp - 1:
            break
    list_score = []
    for name, params in  model.named_modules():
            if isinstance(params, LinearWithSVFT):
                print(name)
                list_score.append(params.get_sorted_list_score())
    
    print("size of list score: ", len(list_score))
    return train_loss / temp            


if __name__ == "__main__":
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
                     
            TestSVFT_scoring(False, args, lora_rank= lora_rank, mode= mode, save_per_ep= save_per_ep, DatasetRoot= root_dataset, log_folder= log_folder, warm_init= warm_init)

    else:
        import determined as det
        from determined.experimental import client
        from determined.pytorch import DataLoader

        info = det.get_cluster_info()
        #args = AttrDict(info.trial.hparams)
        args = SimpleNamespace(**info.trial.hparams)
        args.experiment_id = lora_rank
        if (embedder_ep != None): 
                args.embedder_epochs = embedder_ep
        if (args.embedder_epochs > 0):
            args.finetune_method = args.finetune_method + 'orca' + str(args.embedder_epochs)
        if (mode == 'from_scratch'):
            args.experiment_id = -2
        print("my lora rank: ", lora_rank)
        with det.core.init() as context:
            TestSVFT_scoring(True,args ,info, context, lora_rank= lora_rank, mode = mode, save_per_ep= save_per_ep,DatasetRoot=root_dataset, log_folder= log_folder, warm_init= warm_init)
