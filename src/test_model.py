import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, l2, MMD_loss
from types import SimpleNamespace
from task_configs import get_data, get_config, get_metric, get_optimizer_scheduler, set_trainable, set_grad_state
from newEmbedder import get_src_train_dataset_1Dmodel
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Load CoNLL-2003 dataset


class CustomRoberta(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='roberta', train_epoch=0, activation=None, target_seq_len=512, drop_out=None, from_scratch=False):
        super().__init__()

        self.dense = False
        self.output_raw = True
        self.weight = weight
        self.output_shape = output_shape

        if isinstance(output_shape, tuple):
            self.dense = True
            print("set dense: ", self.dense)

        
        
        modelname = 'roberta-base' 
        self.model = AutoModel.from_pretrained(modelname)
        self.embedder = nn.Identity()

        
        self.model.embeddings = embedder_placeholder()

        
        self.model.pooler = adaptive_pooler()
        self.predictor = nn.Linear(in_features=768, out_features=output_shape)   
        

        if activation == 'sigmoid':
            self.predictor = nn.Sequential(self.predictor, nn.Sigmoid())  
            
        set_grad_state(self.model, False)
        set_grad_state(self.predictor, True)


    def forward(self, x):
        
        if self.output_raw:
            return self.embedder(x) 

        x = self.embedder(x)  
        if x.dim() == 1:  # If shape is (hidden_size,)
           x = x.unsqueeze(0).unsqueeze(0)
             
        x = self.model(inputs_embeds=x)['pooler_output']
        x = self.predictor(x)

        if x.shape[1] == 1 and len(x.shape) == 2:
            x = x.squeeze(1)

        return x

def main(use_determined ,args,info=None, context=None, DatasetRoot= None, log_folder = None):
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
    print("1D task...")
    src_num_classes = 9
    #### get source train dataset 
    print("load src model...")
    src_train_loader = get_src_train_dataset_1Dmodel(args,root)
    src_model = CustomRoberta(sample_shape, src_num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out)
    #################### prepare for train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_model.to(device)
    src_model.train() 
    src_model.output_raw = False
    for name, param in src_model.named_parameters():
       if param.requires_grad:
          print(name)
    print(src_model)      
    
    num_epochs = 30
    optimizer = optim.AdamW(
         src_model.parameters(),
         lr=args.lr if hasattr(args, 'lr') else 1e-4,
         weight_decay=0.05
     )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
         optimizer,
         T_max= num_epochs
     )
    criterion = nn.CrossEntropyLoss(
         label_smoothing=0.1 if hasattr(args, 'label_smoothing') else 0.0  # Optional smoothing
     )
    for epoch in range(num_epochs):
        running_loss = 0.0 
        correct = 0  
        total = 0
        for i, data in enumerate(src_train_loader):
             x_, y_ = data 
             x_ = x_.to(args.device)
             y_ = y_.to(args.device)
             y_ = y_.long() 
             optimizer.zero_grad()
             out = src_model(x_)
             out = F.softmax(out, dim=-1)
             loss = criterion(out, y_)
             loss.backward()
             optimizer.step() 
             running_loss += loss.item()
             _, predicted = torch.max(out, 1)  # Get the index of max log-probability
             total += y_.size(0)
             correct += (predicted == y_).sum().item()

         
        scheduler.step()
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], '
               f'Average Loss: {running_loss/len(src_train_loader):.4f}'
               f' Accuracy: {accuracy:.2f}%')  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORCA')
    parser.add_argument('--config', type=str, default=None, help='config file name')
    parser.add_argument('--embedder_ep', type= int, default= None, help='embedder epoch training')
    parser.add_argument('--root_dataset', type= str, default= None, help='[option]path to customize dataset')
    parser.add_argument('--log_folder', type= str, default= None, help='[option]path to log folder')
    
    args = parser.parse_args()
    embedder_ep = args.embedder_ep
    root_dataset = args.root_dataset
    log_folder = args.log_folder
    if args.config is not None:     
        import yaml
        with open(args.config, 'r') as stream:
            config = yaml.safe_load(stream)
            args = SimpleNamespace(**config['hyperparameters'])
            
            if (embedder_ep != None): 
                args.embedder_epochs = embedder_ep
            if (args.embedder_epochs > 0):
                args.finetune_method = args.finetune_method + 'orca' + str(args.embedder_epochs)
                     
            main(False, args, DatasetRoot= root_dataset, log_folder= log_folder)