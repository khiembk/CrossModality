# Feature-label matching guilded cross-task cross-modality fine-tuning
# 1. Set up env 
# 1.1  Step 1: run  
      startup-hook.sh
# 1.2 Download dataset    
#    1.2.1 NasBench360: 
      Download from link: [NasBench360](https://nb360.ml.cmu.edu/)  
#    1.2.2 PDE:
      Download from link: [PDEDatasets](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) 
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

### Short Explanations on the Config Args:

config: [string] [optional] Name of the configuration file, default = None.

root_dataset: [string] [optional] Path to customize dataset, default = None.

log_folder: [string] [optional] Path to log folder, default = None.

C_entropy: [bool] [optional] Determines if conditional entropy label matching is used, default = False.

second_train: [bool] [optional] Determines if a second training of the model is performed, default = False.

fm_ep: [int] [optional] Number of feature matching epochs, default = None.

lm_ep: [int] [optional] Number of label matching epochs, default = None.

seed: [int] [optional] Seed for training to ensure reproducibility, default = None.

pde: [bool] [optional] Indicates if the dataset is a PDE dataset, default = False.

id: [int] [optional] ID of the experiment, default = None.



      
