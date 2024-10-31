import embedder
from utils import count_params, count_trainable_params, calculate_stats
from lp_embedder import wrapper2DLORA_last,wrapper2DLORA
from embedder import wrapper2D
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
    model = wrapper2DLORA(sample_shape,output_shape,4,classification= False)
    print("trainable params: ", count_trainable_params(model))
    print("all params: ", count_params(model))
    for name, param in model.named_parameters():
       if  param.requires_grad:
           print(f"Layer: {name}")
def main():
    test2D_model()

if __name__ == "__main__":
    main()