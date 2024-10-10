import embedder
from utils import count_params, count_trainable_params, calculate_stats
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

def main():
    test1D_model()

if __name__ == "__main__":
    main()