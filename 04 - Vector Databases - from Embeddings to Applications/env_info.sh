conda create --name dlai-short-courses1 python=3.9.18
conda activate dlai-short-courses1
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install -r requirements.txt
pip install jupyter
python -m ipykernel install --user --name dlai-short-courses1 --display-name "Python (dlai-short-courses1)"
