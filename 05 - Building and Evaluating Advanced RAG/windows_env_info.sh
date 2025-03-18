conda remove -n dlai-short-courses2 --all
conda create -n dlai-short-courses2 python=3.9.18
conda activate dlai-short-courses2
pip install ipykernel
python -m ipykernel install --user --name dlai-short-courses2 --display-name "Python (dlai-short-courses2)"