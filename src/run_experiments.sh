############## NasBench #######################################################################
###1. Darcy 
###1.1 Download dataset.
python DownloadDataset.py --dataset "DARCY-FLOW-5"  
###1.2 Run FLM method.
python newmain.py --config configs/darcy.yaml --seed=0
python newmain.py --config configs/darcy.yaml --seed=1
python newmain.py --config configs/darcy.yaml --seed=2
###1.3 Run ORCA method.
python oldmain.py --config configs/darcy.yaml --seed=0
python oldmain.py --config configs/darcy.yaml --seed=1
python oldmain.py --config configs/darcy.yaml --seed=2
###2.DeepSea.
###2.1 Download dataset.
python DownloadDataset.py --dataset "DEEPSEA"
###2.2 Run FLM method.
python newmain.py --config configs/deepsea.yaml --seed=0
python newmain.py --config configs/deepsea.yaml --seed=1
python newmain.py --config configs/deepsea.yaml --seed=2
###2.3 Run ORCA method.
###2.2 Run FLM method.
python oldmain.py --config configs/deepsea.yaml --seed=0
python oldmain.py --config configs/deepsea.yaml --seed=1
python oldmain.py --config configs/deepsea.yaml --seed=2


