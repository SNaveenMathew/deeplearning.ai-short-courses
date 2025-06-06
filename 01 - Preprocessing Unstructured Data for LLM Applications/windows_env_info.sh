conda create -n dlai-short-courses python=3.9.18
conda activate dlai-short-courses
conda install jupyter
pip install ipykernel
python -m ipykernel install --user --name dlai-short-courses --display-name "Python (dlai-short-courses)"
pip install -r windows_requirements.txt
conda install conda-forge::sentencepiece
pip install spacy
python -m spacy download en_core_web_sm
python setup.py
pip install openllm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
