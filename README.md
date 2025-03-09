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

## Downloading the required files

1. Click View -> File Browser or File -> Open
2. Select multiple files at a time and click 'Download' because it's impossible to install `zip` or any other tool without having admin rights

## Conda environment

To avoid complications create a separate environment to install dependencies. Use multiple environments if required to isolate dependency issues

## Installing dependencies

1. Activate the environment first using 'conda activate <env_name>'. Eg: conda activate dlai-short-courses. Environment name will be listed in the first line of the dependencies.sh file as a comment
2. Copy-paste each line from the respective dependencies.sh file and execute in command-line to install using conda or pip
