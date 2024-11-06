import embedder
from utils import count_params, count_trainable_params, calculate_stats
from lp_embedder import wrapper2DLORA_last,wrapper2DLORA
from embedder import wrapper2D
import numpy as np
import random
import torch
import os 
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
    model = wrapper2DLORA_last(sample_shape,output_shape)
    #print(model)
    print("trainable params: ", count_trainable_params(model))
    print("all params: ", count_params(model))
    for name, param in model.named_parameters():
       if  param.requires_grad:
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

if __name__ == "__main__":
    main()