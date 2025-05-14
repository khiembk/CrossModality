import os
import torch.nn as nn
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn # type: ignore
from timeit import default_timer
from tqdm import tqdm
import yaml
from types import SimpleNamespace
from task_configs import get_data, get_config, get_metric, get_optimizer_scheduler, set_trainable, set_grad_state
from utils import count_params, count_trainable_params, calculate_stats
from newEmbedder import get_pretrain_model2D_feature, wrapper1D, wrapper2D, feature_matching_tgt_model,get_src_train_dataset_1Dmodel
from test_model import get_src_predictor1D
from newEmbedder import label_matching_by_entropy, label_matching_by_conditional_entropy,Embeddings1D, Embeddings2D

class NSAdaptedWrapper(wrapper2D):
    def __init__(self, input_channels=60, output_channels=6, img_size=224 ):
        # Initialize parent with NS-specific parameters
        super().__init__(
            input_shape=(input_channels, img_size, img_size),
            output_shape=(output_channels, img_size, img_size),
            use_embedder=True,
              # Force regression mode
            drop_out=0.1,
            from_scratch=False,
            
        )
        
        # Replace the predictor with NS-specific decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 1)
        )
        
        # Modify embedder for NS data
        self.model.swin.embeddings = NSEmbedder(
            input_shape=(input_channels, img_size, img_size),
            patch_size=4,
            embed_dim=128,
            img_size=img_size,
            config=self.model.config
        )
        
        # Enable gradients for fine-tuning
        set_grad_state(self.model.swin, True)
        set_grad_state(self.decoder, True)

    def forward(self, x):
        # Input shape: [B, 60, 224, 224]
        if self.output_raw:
            return self.model.swin.embeddings(x)[0]
        
        # Process through Swin
        x = self.model(x).logits if hasattr(self.model, 'logits') else self.model(x)
        
        # Reshape if needed (handles both classification and regression outputs)
        if x.dim() == 3:  # [B, seq_len, features]
            B, L, C = x.shape
            H = W = int(L**0.5)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # Process through decoder
        return self.decoder(x)  # Output shape: [B, 6, 224, 224]

class NSEmbedder(Embeddings2D):
    """Custom embedder for Navier-Stokes data"""
    def __init__(self, input_shape, patch_size, embed_dim, img_size, config):
        super().__init__(
            input_shape=input_shape,
            patch_size=patch_size,
            embed_dim=embed_dim,
            img_size=img_size,
            config=config
        )
        
        # Adjust projection for NS input channels
        self.projection = nn.Conv2d(
            input_shape[0],  # Use channel dimension
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Custom initialization for fluid data
        nn.init.kaiming_normal_(self.projection.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.projection.bias)
        
    def forward(self, x):
        # Skip resize if already correct size
        B, C, H, W = x.shape
        
        # Pad if needed
        x = self.maybe_pad(x, H, W)
        
        # Project and flatten
        x = self.projection(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return self.norm(x), self.patched_dimensions
    
def main(use_determined ,args,info=None, context=None, DatasetRoot= None, log_folder = None, second_train = False):
    
    ############## Init log file and set seed
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #args.device = 'cuda' 
    print("The current device is: ", args.device)
    root = '/datasets' if use_determined else './datasets'
    if (DatasetRoot != None):
        if (args.pde):
           root = DatasetRoot
        else:    
           root = DatasetRoot + '/datasets'

    print("Path folder dataset: ",root) 
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed) 
    torch.cuda.manual_seed_all(args.seed)

    
    if args.reproducibility:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
    ##################################################################################################
    print("Test NS dataset.")
    dims, sample_shape, num_classes, loss, args = get_config(root, args)
    print("current configs: ", args)
    
    input_chanels = 60
    output_chanels = 6
    tgt_model = NSAdaptedWrapper(input_channels= input_chanels, output_channels= output_chanels)    
    print("Input chanels: ", input_chanels)
    print("Output chanels: ", output_chanels)
    
    print("final model: ", tgt_model)

    ######### load tgt dataset 
    print("load tgt dataset ...")
    train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split)
    ###########################################################################
    print("<<< Test size of dataSet >>>")
    sample_x, sample_y = train_loader[0]
    print(f"Input shape: {sample_x.shape}")  # Should be (6*10, 224, 224) = (60, 224, 224)
    print(f"Target shape: {sample_y.shape}")  # Should be (6, 224, 224)
    ###########################################################################
    metric, compare_metrics = get_metric(root, args.dataset)
    decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
    ###############################################################################################
    
    print("load dic if model was trained ...")
    tgt_model, ep_start, id_best, train_score, train_losses, embedder_stats_saved = load_state(use_determined, args, context, tgt_model, None, None, n_train, freq=args.validation_freq, test=True)
    # embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    offset = 0 if ep_start == 0 else 1
    args, tgt_model, optimizer, scheduler = get_optimizer_scheduler(args, tgt_model, module=None if args.predictor_epochs == 0 or ep_start >= args.predictor_epochs else 'predictor', n_train=n_train)
    train_full = args.predictor_epochs == 0 or ep_start >= args.predictor_epochs
   
    if args.device == 'cuda':
        tgt_model.cuda()
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
    print("all param count:", count_params(tgt_model),)
   
    print("trainabel params count: %d  ",count_trainable_params(tgt_model))
  
    # print(model)
   
    ###### load state optimizer
    if second_train:
        tgt_model, ep_start, id_best, train_score, train_losses, embedder_statssaved = load_state(use_determined, args, context, tgt_model, None, None, n_train, freq=args.validation_freq, test = True)
    else:
        tgt_model, ep_start, id_best, train_score, train_losses, embedder_statssaved = load_state(use_determined, args, context, tgt_model, optimizer, scheduler, n_train, freq=args.validation_freq)
    #embedder_stats = embedder_stats if embedder_stats_saved is None else embedder_stats_saved
    train_time = []
    embedder_stats = []
    print("\n------- Start Training --------" if ep_start == 0 else "\n------- Resume Training --------")

    for ep in range(ep_start, args.epochs + args.predictor_epochs):
        if not train_full and ep >= args.predictor_epochs:
            args, tgt_model, optimizer, scheduler = get_optimizer_scheduler(args, tgt_model, module=None, n_train=n_train)
            train_full = True

        time_start = default_timer()

        train_loss = train_one_epoch(context, args, tgt_model, optimizer, scheduler, train_loader, loss, n_train, decoder, transform)
        train_time_ep = default_timer() -  time_start 

        if ep % args.validation_freq == 0 or ep == args.epochs + args.predictor_epochs - 1: 
                
            val_loss, val_score = evaluate(context, args, tgt_model, val_loader, loss, metric, n_val, decoder, transform, fsd_epoch=ep if args.dataset == 'FSD' else None)

            train_losses.append(train_loss)
            train_score.append(val_score)
            train_time.append(train_time_ep)

            print("[train", "full" if ep >= args.predictor_epochs else "predictor", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))
           
            if use_determined :
                id_current = save_state(use_determined, args, context, tgt_model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats)
                try:
                    context.train.report_training_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"train loss": train_loss, "epoch time": train_time_ep})
                    context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train + offset, metrics={"val score": val_score})
                except:
                    pass
                    
            if compare_metrics(train_score) == val_score:
                if not use_determined :
                    print("save state at epoch ep: ", ep)
                    id_current = save_state(use_determined, args, context, tgt_model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats)
                id_best = id_current
            

        if ep == args.epochs + args.predictor_epochs - 1:
            print("\n------- Start Test --------")
            test_scores = []
            test_model = tgt_model
            test_time_start = default_timer()
            test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            test_time_end = default_timer()
            test_scores.append(test_score)

            print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)

            test_model, _, _, _, _, _ = load_state(use_determined, args, context, test_model, optimizer, scheduler, n_train, id_best, test=True)
            test_time_start = default_timer()
            test_loss, test_score = evaluate(context, args, test_model, test_loader, loss, metric, n_test, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
            test_time_end = default_timer()
            test_scores.append(test_score)

            print("[test best-validated]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
            

            if use_determined:
                checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
                with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
                    np.save(os.path.join(path, 'test_score.npy'), test_scores)
            else:
                path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
                np.save(os.path.join(path, 'test_score.npy'), test_scores)

           
        if use_determined and context.preempt.should_preempt():
            print("paused")
            return

   
       


def linear_probing(args, model, train_loader, val_loader, test_loader, metric, compare_metrics,decoder,transform, Roberta, loss,n_train,n_val,linear_prob_ep=5):
    ###### check model
    print("Freeze body model...")
    if Roberta:
            print("Freeze 1D bodymodel...")
            set_grad_state(model.model.encoder,False)
            set_grad_state(model.predictor, True)
    else:
            print("Freeze 2D body model...")
            set_grad_state(model.model.swin.encoder,False)
            set_grad_state(model.predictor, True)   
    ###### load optimizer
    args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, n_train=n_train)
    if args.device == 'cuda':
        model.cuda()
        try:
            loss.cuda()
        except:
            pass
        if decoder is not None:
            decoder.cuda()

    ###### start linear probing
    print("\n------- Start Linear Probing --------" )
    train_losses = []
    train_score = []
    train_time = [] 
    for ep in range(linear_prob_ep):
        ##### train
        time_start = default_timer()
        train_loss = train_one_epoch(None, args, model, optimizer, scheduler, train_loader, loss, n_train, decoder, transform)
        train_time_ep = default_timer() -  time_start 
        #### eval
        val_loss, val_score = evaluate(None, args, model, val_loader, loss, metric, n_val, decoder, transform, fsd_epoch=ep if args.dataset == 'FSD' else None)
        train_losses.append(train_loss)
        train_score.append(val_score)
        train_time.append(train_time_ep)
        print("[train", "predictor" , ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))
    
    print("\n------- Finish Linear Probing --------" )
    ###### set full model trainable
    print("Set all params trainable...")
    set_grad_state(model,True) 
    ###### delete trash
    return model

def train_one_epoch(context, args, model, optimizer, scheduler, loader, loss, temp, decoder=None, transform=None):    

    model.train()             
    train_loss = 0
    optimizer.zero_grad()

    for i, data in enumerate(tqdm(loader, desc="Training Progress", leave=True)):

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

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if (i + 1) % args.accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if args.lr_sched_iter:
            scheduler.step()

        train_loss += l.item()
        
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


########################## Helper Funcs ##########################

def save_state(use_determined, args, context, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        if not os.path.exists(path):
            os.makedirs(path)
        
        save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats)
        return ep

    else:
        checkpoint_metadata = {"steps_completed": (ep + 1) * n_train, "epochs": ep}
        with context.checkpoint.store_path(checkpoint_metadata) as (path, uuid):
            save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats)
            return uuid


def save_with_path(path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats):
    np.save(os.path.join(path, 'hparams.npy'), args)
    np.save(os.path.join(path, 'train_score.npy'), train_score)
    np.save(os.path.join(path, 'train_losses.npy'), train_losses)
    np.save(os.path.join(path, 'embedder_stats.npy'), embedder_stats)

    model_state_dict = {
                'network_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
    torch.save(model_state_dict, os.path.join(path, 'state_dict.pt'))

    rng_state_dict = {
                'cpu_rng_state': torch.get_rng_state(),
                'gpu_rng_state': torch.get_rng_state(),
                # work around for new server
                'numpy_rng_state': np.random.get_state(),
                'py_rng_state': random.getstate()
            }
    torch.save(rng_state_dict, os.path.join(path, 'rng_state.ckpt'))


def load_embedder(use_determined, args):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        return os.path.isfile(os.path.join(path, 'state_dict.pt'))
    else:

        info = det.get_cluster_info()
        checkpoint_id = info.latest_checkpoint
        return checkpoint_id is not None

def check_if_continue_training(args):
    """Check if have a check point from result.

    Args:
        args (_type_): _description_

    Returns:
        boolean 
    """
    path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
    if not os.path.isfile(os.path.join(path, 'state_dict.pt')):
            return False
    return True
        
def load_state(use_determined, args, context, model, optimizer, scheduler, n_train, checkpoint_id=None, test=False, freq=1):
    if not use_determined:
        path = 'results/'  + args.dataset +'/' + str(args.finetune_method) + '_' + str(args.experiment_id) + "/" + str(args.seed)
        if not os.path.isfile(os.path.join(path, 'state_dict.pt')):
            return model, 0, 0, [], [], None
    else:

        if checkpoint_id is None:
            info = det.get_cluster_info()
            checkpoint_id = info.latest_checkpoint
            if checkpoint_id is None:
                return model, 0, 0, [], [], None
        
        checkpoint = client.get_checkpoint(checkpoint_id)
        path = checkpoint.download()

    train_score = np.load(os.path.join(path, 'train_score.npy'))
    train_losses = np.load(os.path.join(path, 'train_losses.npy'))
    embedder_stats = np.load(os.path.join(path, 'embedder_stats.npy'))
    epochs = freq * (len(train_score) - 1) + 1
    checkpoint_id = checkpoint_id if use_determined else epochs - 1
    model_state_dict = torch.load(os.path.join(path, 'state_dict.pt'))
    model.load_state_dict(model_state_dict['network_state_dict'])
    
    if not test:
        optimizer.load_state_dict(model_state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(model_state_dict['scheduler_state_dict'])

        rng_state_dict = torch.load(os.path.join(path, 'rng_state.ckpt'), map_location='cpu')
        # torch.set_rng_state(rng_state_dict['cpu_rng_state'])
        # torch.cuda.set_rng_state(rng_state_dict['gpu_rng_state'])
        # np.random.set_state(rng_state_dict['numpy_rng_state'])
        # random.setstate(rng_state_dict['py_rng_state'])

        if use_determined: 
            try:
                for ep in range(epochs):
                    if ep % freq == 0:
                        context.train.report_training_metrics(steps_completed=(ep + 1) * n_train, metrics={"train loss": train_losses[ep // freq]})
                        context.train.report_validation_metrics(steps_completed=(ep + 1) * n_train, metrics={"val score": train_score[ep // freq]})
            except:
                print("load error")

    return model, epochs, checkpoint_id, list(train_score), list(train_losses), embedder_stats



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORCA')
    parser.add_argument('--config', type=str, default=None, help='config file name')
    parser.add_argument('--root_dataset', type= str, default= None, help='[option]path to customize dataset')
    parser.add_argument('--log_folder', type= str, default= None, help='[option]path to log folder')
    parser.add_argument('--C_entropy', type= bool, default= False, help='[option]determind Conditional entropy label matching or not')
    parser.add_argument('--freeze_bodymodel', type= bool, default= False, help='[option]determind freeze_body model or not')
    parser.add_argument('--second_train', type= bool, default= False, help='[option]determind second train model or not')
    parser.add_argument('--lp_ep', type= int, default= None, help='Number of linear probing')
    parser.add_argument('--fm_ep', type= int, default= None, help='Number of feature matching')
    parser.add_argument('--lm_ep', type= int, default= None, help='Number of label matching')
    parser.add_argument('--pde', type= bool, default= False, help='[optional]PDE dataset or not')
    args = parser.parse_args()
    fm_ep = args.fm_ep
    lm_ep = args.lm_ep
    root_dataset = args.root_dataset
    log_folder = args.log_folder
    C_entropy = args.C_entropy
    second_train = args.second_train
    freeze_bodymodel = args.freeze_bodymodel
    lp_ep = args.lp_ep
    pde = args.pde
    ############################################
    if args.config is not None:     
        import yaml
        with open(args.config, 'r') as stream:
            config = yaml.safe_load(stream)
            args = SimpleNamespace(**config['hyperparameters'])
            
            if (fm_ep != None): 
                args.embedder_epochs = fm_ep
            if (lm_ep != None):
                args.label_epochs = lm_ep    
            if (args.embedder_epochs > 0):
                args.finetune_method = args.finetune_method + 'FM_CE' + str(args.embedder_epochs)
            if (freeze_bodymodel):
                args.finetune_method = args.finetune_method + '_freeze_bodymodel'
            ################################################################    
            setattr(args, 'freeze_bodymodel', freeze_bodymodel)    
            setattr(args, 'C_entropy', C_entropy)
            setattr(args, 'lp_ep', lp_ep)
            setattr(args, 'pde', pde)
            if not hasattr(args, 'label_epochs'):
                setattr(args, 'label_epochs', 1)         
            main(False, args, DatasetRoot= root_dataset, log_folder= log_folder, second_train= second_train)
    else:
        print("Config for training not found...")
    