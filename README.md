# Feature-label matching guilded cross-task cross-modality fine-tuning
# 1. Set up env 
#   Step 1: run  
      startup-hook.sh

# 2. Run code on NasBench360
# 2.1 Run feature-label matching method (only config required): 
      python newmain.py --configs/task.yaml --log_folder "Logfolder" --root_dataset "pathToDataSet"  
# 2.2 Reproduce ORCA (only config required): 
      python oldmain.py --configs/task.yaml --log_folder "Logfolder" --root_dataset "pathToDataSet" 

# 3. Run code on PDEBench
# 3.1 Run feature-label matching method (only config required): 
      python newmain.py --configs/task.yaml --log_folder "Logfolder" --root_dataset "pathToDataSet"  --pde= True
# 3.2 Reproduce ORCA (only config required): 
      python oldmain.py --configs/task.yaml --log_folder "Logfolder" --root_dataset "pathToDataSet"  --pde= True

      
