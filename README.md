# CrossModality
# 1. Set up env 
#   Step 1: run  
      startup-hook.sh

# 2. Run code
# 2.1 Run feature-label matching method (only config required): 
      python newmain.py --configs/task.yaml --log_folder "Logfolder" --root_dataset "pathToDataSet"  
# 2.2 Reproduce ORCA (only config required): 
      python oldmain.py --configs/task.yaml --log_folder "Logfolder" --root_dataset "pathToDataSet" 

      
