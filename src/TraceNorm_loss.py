import torch.nn as nn
import os
import argparse
import random
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn # type: ignore
from timeit import default_timer
from tqdm import tqdm
import yaml
from types import SimpleNamespace
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling, RobertaForTokenClassification
from lp_embedder import Embeddings2D
from task_configs import get_data, get_config, get_metric, get_optimizer_scheduler, set_trainable
from utils import count_params, count_trainable_params, calculate_stats
from utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, l2, MMD_loss

def custom_loss_function(model, pre_trained_weights, outputs, targets, main_loss_fn, reg_lambda=0.1):
    """
    Custom loss function for Swin-base Vision Transformer with trace norm regularization.
    
    Args:
        model (nn.Module): The current model.
        pre_trained_weights (dict): Pre-trained weights as a dictionary.
        outputs (torch.Tensor): Model outputs.
        targets (torch.Tensor): Ground truth targets.
        main_loss_fn (callable): Main loss function, e.g., nn.CrossEntropyLoss.
        reg_lambda (float): Regularization strength.
    
    Returns:
        torch.Tensor: Total loss (main loss + regularization loss).
    """
    # Compute the main loss
    main_loss = main_loss_fn(outputs, targets)
    
    # Initialize regularization loss
    reg_loss = 0.0

    # Iterate over the model's named parameters
    for name, param in model.named_parameters():
        if name in pre_trained_weights and param.requires_grad:  # Ensure it's trainable
            pre_trained_param = pre_trained_weights[name]
            pre_trained_param = pre_trained_param.to(param.device)
     
            # Compute the difference
            diff = param - pre_trained_param
            if diff.ndim == 2:
            # Add the trace norm (sum of singular values)
               reg_loss += torch.norm(diff, p='nuc')  # Trace norm = nuclear norm

    # Combine main loss and regularization loss
    total_loss = main_loss + reg_lambda * reg_loss
    return total_loss



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

        if use_embedder:
            self.embedder = Embeddings2D(input_shape, patch_size=patch_size, config=self.model.config, embed_dim=embed_dim, img_size=img_size)
            #set_grad_state(self.embedder, True)
            self.model.swin.embeddings = self.embedder  

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

def main(use_determined ,args,info=None, context=None, lora_rank=1, mode = 'lora', save_per_ep = 1, DatasetRoot= None, log_folder = None, warm_init = True):
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

    log_file = f"lora{args.experiment_id}_{args.dataset}_{args.finetune_method}.log"
    if (log_folder is not None):
        log_dir = os.path.join(log_folder)
        os.makedirs(log_dir, exist_ok= True)
        log_file = os.path.join(log_dir, f"lora{args.experiment_id}_{args.dataset}_{args.finetune_method}.log")

    logging.basicConfig(filename= log_file,
                    level=logging.INFO,  # Set logging level
                    format='%(asctime)s - %(levelname)s - %(message)s') 
    if args.reproducibility:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
    
    dims, sample_shape, num_classes, loss, args = get_config(root, args)
    ### call model 
    #wrapper_func = wrapper1D if len(sample_shape) == 3 else wrapper2D
    pretrain_model = wrapper2D_pretrain(sample_shape, num_classes,weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
    model = wrapper2D_pretrain(sample_shape, num_classes,weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
    for param in pretrain_model.parameters():
        param.requires_grad = False
    pre_trained_weights = {name: param.clone().detach() for name, param in pretrain_model.named_parameters()}
    model.output_raw = False
    model.train_predictor = False
    model.train_embedder = False
    print("first call model : ")
    print("all param count:", count_params(model))
    print("trainabel params count :  ",count_trainable_params(model))    
    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split)
    metric, compare_metrics = get_metric(root, args.dataset)
    decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
    ep_start = 0
    offset = 0 if ep_start == 0 else 1
    print("before get optimizer scheduler : ")
    print("all param count:", count_params(model))
    print("trainabel params count :  ",count_trainable_params(model)) 
    args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None if args.predictor_epochs == 0 or ep_start >= args.predictor_epochs else 'predictor', n_train=n_train)
    print("affter get optimizer scheduler : ")
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
    logging.info("all param count: %d", count_params(model))
    print("trainabel params count: %d  ",count_trainable_params(model))
    logging.info("trainabel params count:  %d  ",count_trainable_params(model))
    print("print model")
    print(model)
    logging.info(f"Model Structure:\n{model}")
    embedder_stats = []
    train_time = []
    ep_start = 0
    train_losses = []
    train_score = []
    print("\n------- Start Training --------" if ep_start == 0 else "\n------- Resume Training --------")

    for ep in range(ep_start, args.epochs + args.predictor_epochs):
        if not train_full and ep >= args.predictor_epochs:
            args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None, n_train=n_train)
            train_full = True

        time_start = default_timer()
        train_loss = train_one_epoch(context, args, model, optimizer, scheduler, train_loader, loss, n_train, pre_trained_weights, decoder=decoder, transform=transform)

        train_time_ep = default_timer() -  time_start 

        if ep % args.validation_freq == 0 or ep == args.epochs + args.predictor_epochs - 1: 
                
            val_loss, val_score = evaluate(context, args, model, val_loader, loss, metric, n_val, decoder, transform, fsd_epoch=ep if args.dataset == 'FSD' else None)

            train_losses.append(train_loss)
            train_score.append(val_score)
            train_time.append(train_time_ep)

            print("[train", "full" if ep >= args.predictor_epochs else "predictor", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))
            logging.info(
            "[train %s %d %.6f] time elapsed: %.4f\ttrain loss: %.4f\tval loss: %.4f\tval score: %.4f\tbest val score: %.4f",
            "full" if ep >= args.predictor_epochs else "predictor",ep,optimizer.param_groups[0]['lr'], train_time[-1], train_loss,
             val_loss, val_score,compare_metrics(train_score))
            if use_determined :
                if ep % save_per_ep ==0 :
                   print("save state at epoch ep: ", ep)
                try:
                    context.train.report_training_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"train loss": train_loss, "epoch time": train_time_ep})
                    context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"val score": val_score})
                except:
                    pass
                    
            
        if ep == args.epochs + args.predictor_epochs - 1:
            print("\n------- Start Test --------")
            test_scores = []
            test_model = model
            test_time_start = default_timer()
            test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            test_time_end = default_timer()
            test_scores.append(test_score)

            print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            logging.info("[test last]\ttime elapsed: %.4f\ttest loss: %.4f\ttest score: %.4f",test_time_end - test_time_start,test_loss,test_score)

            test_time_start = default_timer()
            test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            test_time_end = default_timer()
            test_scores.append(test_score)

            print("[test best-validated]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            logging.info("[test best-validated]\ttime elapsed: %.4f\ttest loss: %.4f\ttest score: %.4f" % (test_time_end - test_time_start, test_loss, test_score))
            if (mode == "ada"):
                logging.info(f"Affter fix Structure:\n{model}")

            if use_determined:
                checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
                # with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
                #     np.save(os.path.join(path, 'test_score.npy'), test_scores)
            else:
                path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
                # np.save(os.path.join(path, 'test_score.npy'), test_scores)

           
        if use_determined and context.preempt.should_preempt():
            print("paused")
            return
    
    model.get_differnt_rank_all_layer_with_model(model= model, model1 = pretrain_model)

def train_one_epoch(context, args, model, optimizer, scheduler, loader, loss, temp, pre_train_weight ,decoder=None, transform=None, mode = 'lora'):    

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
        
        l = custom_loss_function(model= model, pre_trained_weights= pre_train_weight,outputs= out,targets= y,main_loss_fn= loss)
        l.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if (i + 1) % args.accum == 0:
            optimizer.step()
            if (mode == 'ada') :
                model.model.update_and_allocate(i)
            optimizer.zero_grad()
        
        if args.lr_sched_iter:
            scheduler.step()

        train_loss += l.item()
        print(f'Batch [{i+1}/{len(loader)}], Loss: {l.item():.4f}')
        if i >= temp - 1:
            break

    if (not args.lr_sched_iter):
        scheduler.step()

    return train_loss / temp


def evaluate(context, args, model, loader, loss, metric, n_eval, decoder=None, transform=None, fsd_epoch=None):
    model.eval()
    
    eval_loss, eval_score = 0, 0
    
    if fsd_epoch is None:

        ys, outs, n_eval, n_data = [], [], 0, 0

        with torch.no_grad():
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

                outs.append(out)
                ys.append(y)
                n_data += x.shape[0]

                if n_data >= args.eval_batch_size or i == len(loader) - 1:
                    outs = torch.cat(outs, 0)
                    ys = torch.cat(ys, 0)

                    eval_loss += loss(outs, ys).item()
                    eval_score += metric(outs, ys).item()
                    n_eval += 1

                    ys, outs, n_data = [], [], 0

            eval_loss /= n_eval
            eval_score /= n_eval

    else:
        outs, ys = [], []
        with torch.no_grad():
            for ix in range(loader.len):

                x, y = loader[ix]
                x, y = x.to(args.device), y.to(args.device)
                out = model(x).mean(0).unsqueeze(0)
                eval_loss += loss(out, y).item()
                outs.append(torch.sigmoid(out).detach().cpu().numpy()[0])
                ys.append(y.detach().cpu().numpy()[0])

        outs = np.asarray(outs).astype('float32')
        ys = np.asarray(ys).astype('int32')
        stats = calculate_stats(outs, ys)
        eval_score = 1-np.mean([stat['AP'] for stat in stats])
        eval_loss /= n_eval

    return eval_loss, eval_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORCA')
    parser.add_argument('--config', type=str, default=None, help='config file name')
    parser.add_argument('--lora_rank', type= int, default= -1, help='LORA rank')
    parser.add_argument('--mode', type= str, default= 'lora', help='mode for ada or lora')
    parser.add_argument('--embedder_ep', type= int, default= None, help='embedder epoch training')
    parser.add_argument('--ep', type= int, default= None, help='epoch training')
    parser.add_argument('--save_per_ep', type= int, default= 1, help='save per epoch')
    parser.add_argument('--root_dataset', type= str, default= None, help='[option]path to customize dataset')
    parser.add_argument('--log_folder', type= str, default= None, help='[option]path to log folder')
    parser.add_argument('--warm_init', type= bool, default= True, help='warm init controller')
    args = parser.parse_args()
    lora_rank = args.lora_rank
    embedder_ep = args.embedder_ep
    save_per_ep = args.save_per_ep
    mode = args.mode 
    ep = args.ep
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
            if ep is not None:
                args.epochs = ep         
            main(False, args, lora_rank= lora_rank, mode= mode, save_per_ep= save_per_ep, DatasetRoot= root_dataset, log_folder= log_folder, warm_init= warm_init)

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
            main(True,args ,info, context, lora_rank= lora_rank, mode = mode, save_per_ep= save_per_ep,DatasetRoot=root_dataset, log_folder= log_folder, warm_init= warm_init)


def custom_loss_function(model, pre_trained_weights, outputs, targets, main_loss_fn, reg_lambda=0.1):
    """
    Custom loss function for Swin-base Vision Transformer with trace norm regularization.
    
    Args:
        model (nn.Module): The current model.
        pre_trained_weights (dict): Pre-trained weights as a dictionary.
        outputs (torch.Tensor): Model outputs.
        targets (torch.Tensor): Ground truth targets.
        main_loss_fn (callable): Main loss function, e.g., nn.CrossEntropyLoss.
        reg_lambda (float): Regularization strength.
    
    Returns:
        torch.Tensor: Total loss (main loss + regularization loss).
    """
    # Compute the main loss
    main_loss = main_loss_fn(outputs, targets)

    # Initialize regularization loss
    reg_loss = 0.0

    # Iterate over the model's named parameters
    for name, param in model.named_parameters():
        if name in pre_trained_weights and param.requires_grad:  # Ensure it's trainable
            pre_trained_param = pre_trained_weights[name]
            # Compute the difference
            diff = param - pre_trained_param
            # Add the trace norm (sum of singular values)
            reg_loss += torch.norm(diff, p='nuc')  # Trace norm = nuclear norm

    # Combine main loss and regularization loss
    total_loss = main_loss + reg_lambda * reg_loss
    return total_loss


