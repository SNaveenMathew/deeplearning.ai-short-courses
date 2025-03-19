conda remove -n dlai-short-courses2 --all
conda env create -f windows_environment.yaml
conda activate dlai-short-courses2
pip install ipykernel
python -m ipykernel install --user --name dlai-short-courses2 --display-name "Python (dlai-short-courses2)"
# pip install "trulens-apps-llamaindex>=1.0.0"
# pip install "trulens-providers-openai>=1.0.0"
# pip install llama_index[langchain]