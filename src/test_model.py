import embedder
from utils import count_params, count_trainable_params, calculate_stats
from lp_embedder import wrapper2DLORA_last,wrapper2DLORA
from embedder import wrapper2D
import numpy as np
import random
import torch
import os 
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling, RobertaForTokenClassification
import torch.nn as nn
from lp_embedder import Embeddings2D
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
    output_shape = 3
    model = wrapper2D_pretrain(sample_shape,output_shape)
    model1 = wrapper2D_pretrain(sample_shape,output_shape)
    model.get_differnt_rank_all_layer_with_model(model, model1)
    #print(model)
    # print("trainable params: ", count_trainable_params(model))
    # print("all params: ", count_params(model))
    # for name, param in model.named_parameters():
    #    if  param.requires_grad:
    #        print(f"Layer: {name}")
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

if __name__ == "__main__":
    main()