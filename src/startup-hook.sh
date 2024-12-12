CURPATH=`pwd`

# OTDD 
pip install gdown
cd otdd
pip install -r requirements.txt
pip install .
pip install torch-scatter torch-spars -f https://data.pyg.org/whl/torch-2.4.1%2Bcu124.html
# for FSD50K
# pip install librosa
# apt-get install -y libsndfile1

cd $CURPATH

