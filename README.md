# deeplearning.ai-short-courses
Space for local runs and experiments with deeplearning.ai short courses

## Disclaimer

Almost all the code is COPIED. I do not claim intellectual rights over the code or any assets. I intend to use the code shared by the respective creators to satisfy my curiosity and/or to build something useful.

## Honest short course ratings

Note: This is my opinion on the courses. This may not reflect their true ratings.

| Course name | Prerequisites | Effectiveness in projects | Usefulness | Difficulty (5 = tough, 0 = easy) | Comments |
| -------- | ------- | ------- | ------- | ------- | ------- |
| [Preprocessing Unstructured Data for LLM Applications](https://learn.deeplearning.ai/courses/preprocessing-unstructured-data-for-llm-applications) | Python | ![stars](https://starrating-beta.vercel.app/3.5/) | ![stars](https://starrating-beta.vercel.app/2.5/)  | ![stars](https://starrating-beta.vercel.app/2.5/) | Using the `unstructured` API is very painful, especially since it's almost at EOL. Didn't work even after receiving an email from Unstructured Marketing Team |
| [Attention in Transformers: Concepts and Code in PyTorch](https://learn.deeplearning.ai/courses/attention-in-transformers-concepts-and-code-in-pytorch) | Python, PyTorch, Linear Algebra | ![stars](https://starrating-beta.vercel.app/1.5/) | ![stars](https://starrating-beta.vercel.app/3.5/)  | ![stars](https://starrating-beta.vercel.app/3.0/) | Theoretical course. Very useful for attention-based model training and debugging, but kinda hard to find a use if your mentality is "build at any cost even if I don't undertand it" |
| [LangChain: Chat with your Data](https://learn.deeplearning.ai/courses/langchain-chat-with-your-data/) | Python | ![stars](https://starrating-beta.vercel.app/4.5/) | ![stars](https://starrating-beta.vercel.app/4.25/)  | ![stars](https://starrating-beta.vercel.app/2.0/) | This is already an end-to-end course. Following-up with some RAG metrics, instruction tuning and app building (probably not Gradio) will cover most common LLM based goals |
| [Vector Databases: from Embeddings to Applications](https://learn.deeplearning.ai/courses/vector-databases-embeddings-applications/) | Python | ![stars](https://starrating-beta.vercel.app/0.25/) | ![stars](https://starrating-beta.vercel.app/1.0/)  | ![stars](https://starrating-beta.vercel.app/2.0/) | Frankly this course is supposed to be amazing. But Weaviate makes it 1000 times worse than it should be. Configuring the environment was a pain. Worst of all, Weaviate doesn't have a free tier - therefore, nothing works without paying. |
| [Building and Evaluating Advanced RAG](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/) | Python | ![stars](https://starrating-beta.vercel.app/4.25/) | ![stars](https://starrating-beta.vercel.app/4.0/)  | ![stars](https://starrating-beta.vercel.app/4.0/) | Very useful course, but most of the OpenAI related stuff won't work without adding a payment method. Need to find a suitable open-source replacement for OpenAI. |
| [How Diffusion Models Work](https://learn.deeplearning.ai/courses/diffusion-models/) | Python, some math/stat | ![stars](https://starrating-beta.vercel.app/3.75/) | ![stars](https://starrating-beta.vercel.app/4.2/)  | ![stars](https://starrating-beta.vercel.app/4.3/) | Hands on - easy to get the base version working; need to experiment with LLM embeddings. Theory - tough to understand a few aspects; like why denoise, then add noise (otherwise there's mode collapse)? Need to understand how sampling methods DDPM and DDIM work. |

## Downloading the required files

1. Click View -> File Browser or File -> Open
2. Select multiple files at a time and click 'Download' because it's impossible to install `zip` or any other tool without having admin rights

## Conda environment

To avoid complications create a separate environment to install dependencies. Use multiple environments if required to isolate dependency issues

## Installing dependencies

1. Activate the environment first using 'conda activate <env_name>'. Eg: conda activate dlai-short-courses
2. Run `pip install -r requirements.txt` in the order in which the courses are listed
3. Some notebooks mention installation of specific packages+versions - follow the installation steps in the same order as the courses and notebooks (both in ascending order) to avoid breaking the dependencies in the conda environment (as of course 3 all the newly installed packages were back-tested and found to be working without errors)
