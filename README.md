# CrossModality
# 1. Set up env 
#   Step 1: run  
      startup-hook.sh
#   Step 2: run 
      pip install peft
# 2. Run code
#  2.1 nomal-ORCA fine-tune: 
       python main.py --config configs/task.yaml --embedder_ep=x   --root_dataset path/parent/of/dataset
      #run ORCA only need to set embedder epoch >0  in task.yaml
      # x: embedder epochs, it should be 0 or 60
#  2.2 LoRA fine-tune:  
      python main.py --configs/task.yaml --lora_rank=r --embedder_ep=x  --root_dataset path/parent/of/dataset
      #r(int) = lora rank
      # x: embedder epochs, it should be 0 or 60
#  2.3 Adaptive LORA fine-tune: 
       python main.py --configs/task.yaml --lora_rank=r --mode ada --embedder_ep=x --root_dataset path/parent/of/dataset
       #r(int) = average lora rank. 
       # x: embedder epochs, it should be 0 or 60    
#  2.4 Train from scratch: 
       python main.py --config configs/task.yaml --mode from_scratch --root_dataset path/parent/of/dataset
# 3. Hyper-parameter:
   parser = argparse.ArgumentParser(description='ORCA')
   parser.add_argument('--config', type=str, default=None, help='config file name')
   parser.add_argument('--lora_rank', type= int, default= -1, help='LORA rank')
   parser.add_argument('--mode', type= str, default= 'lora', help='mode for ada or lora')
   parser.add_argument('--embedder_ep', type= int, default= None, help='embedder epoch training')
   parser.add_argument('--save_per_ep', type= int, default= 1, help='save per epoch')
   parser.add_argument('--root_dataset', type= str, default= None, help='[option]path to customize dataset')
   parser.add_argument('--log_folder', type= str, default= None, help='[option]path to log folder')
