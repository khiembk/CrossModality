# CrossModality
# 1. Set up env 
#   Step 1: run  
      startup-hook.sh

# 2. Run code
# 2.1 Run LORA with p (1<=p <=6)
      python main.py --configs/task.yaml --p=x --log_folder "Logfolder" --root_dataset "pathToDataSet"  
