# CrossModality
# 1. Set up env 
#   Step 1: run startup-hook.sh
#   Step 2: run pip install peft
# 2. Run code
#  2.1 nomal-ORCA fine-tune: python main.py --config configs/task.yaml   
      # run ORCA only need to set embedder epoch >0  in task.yaml
#  2.2 LoRA fine-tune:  python main.py --configs/task.yaml --lora_rank=r 
      # r(int) = lora rank
#  2.3 Adaptive LORA fine-tune: python main.py --configs/task.yaml --lora_rank=r --mode ada
      # r(int) = average lora rank.     
#  2.4 Train from scratch: python main.py --config configs/task.yaml --mode from_scratch 