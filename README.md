# CrossModality
# 1. Set up env 
#   Step 1: run  
      startup-hook.sh
#   Step 2: run 
      pip install peft
# 2. Run code
#  2.1 nomal-ORCA fine-tune: 
       python main.py --config configs/task.yaml --embedder_ep=x
      #run ORCA only need to set embedder epoch >0  in task.yaml
      # x: embedder epochs, it should be 0 or 60
#  2.2 LoRA fine-tune:  
      python main.py --configs/task.yaml --lora_rank=r --embedder_ep=x
      #r(int) = lora rank
      # x: embedder epochs, it should be 0 or 60
#  2.3 Adaptive LORA fine-tune: 
       python main.py --configs/task.yaml --lora_rank=r --mode ada --embedder_ep=x
       #r(int) = average lora rank. 
       # x: embedder epochs, it should be 0 or 60    
#  2.4 Train from scratch: 
       python main.py --config configs/task.yaml --mode from_scratch 