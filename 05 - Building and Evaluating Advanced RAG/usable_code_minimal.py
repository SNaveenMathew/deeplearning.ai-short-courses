import utils
import numpy as np
import pandas as pd
import os
# import openai
import nest_asyncio
from copy import deepcopy

from utils import build_automerging_index
from utils import get_automerging_query_engine
from utils import get_prebuilt_trulens_recorder
from utils import get_sentence_window_query_engine
from utils import build_sentence_window_index

from llama_index.core import Document
from llama_index.core import load_index_from_storage
from llama_index.core import QueryBundle
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import SentenceWindowNodeParser
# from llama_index.core.response.notebook_utils import display_response# Use this in Jupyter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.schema import TextNode
from llama_index.llms.ollama import Ollama
# from llama_index.llms.openai import OpenAI

from trulens_eval import Feedback
# from trulens_eval import OpenAI as fOpenAI
from trulens_eval import Tru
from trulens_eval import TruLlama
from trulens_eval.feedback import LiteLLM

# openai.api_key = utils.get_openai_api_key()

documents = SimpleDirectoryReader(
    input_dir="../../tech_non_tech_blog/_posts/"
).load_data()

print(type(documents), "\n")
print(len(documents), "\n")
print(type(documents[0]))
print(documents[0])


# Basic RAG pipeline
document = Document(text="\n\n".join([doc.text for doc in documents]))
llm = Ollama(model="llama3.2:1b", request_timeout=60.0)# LLMs that can run locally: deepseek-r1

from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
# service_context = ServiceContext.from_defaults(
#     llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
# )
index = VectorStoreIndex.from_documents([document],
                                        # service_context=service_context)
                                        settings=Settings)

query_engine = index.as_query_engine()
response = query_engine.query(
    "What is my laptop configuration?"
)
print(str(response))


# Evaluation setup using TruLens
eval_questions = [
    "What is my laptop configuration?",
    "How long will it take for brute force to complete?"
]
tru = Tru()
tru.reset_database()
tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                             app_id="Direct Query Engine")

with tru_recorder as recording:
    for question in eval_questions:
        response = query_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()
# launches on http://localhost:8501/
tru.run_dashboard()


# Advanced RAG pipeline
## 1. Sentence Window retrieval
# llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)
sentence_window_engine = get_sentence_window_query_engine(sentence_index)
window_response = sentence_window_engine.query(
    "How to speed up diff between consecutive rows in python?"
)
print(str(window_response))

tru.reset_database()
tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id = "Sentence Window Query Engine"
)

for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))

tru.get_leaderboard(app_ids=[])
# launches on http://localhost:8501/
tru.run_dashboard()


## 2. Auto-merging retrieval
automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)
automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)
auto_merging_response = automerging_query_engine.query(
    eval_questions[0]
)
print(str(auto_merging_response))

tru.reset_database()
tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                                                         app_id="Automerging Query Engine")

for question in eval_questions:
    with tru_recorder_automerging as recording:
        response = automerging_query_engine.query(question)
        print(question)
        print(response)

tru.get_leaderboard(app_ids=[])
# launches on http://localhost:8501/
tru.run_dashboard()


## Feedback functions
nest_asyncio.apply()
# provider = fOpenAI()
provider = LiteLLM(
    model_engine=f"ollama/llama3.2:1b", 
    endpoint="http://localhost:11435"
)

### 1. Answer Relevance
f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()


### 2. Context Relevance
context_selection = TruLlama.select_source_nodes().node.text
f_qs_relevance = (
    Feedback(provider.qs_relevance,
             name="Context Relevance")
    .on_input()
    .on(context_selection)
    .aggregate(np.mean)
)


### 3. Groundedness
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
)


## Evaluation of the RAG application
tru_recorder = TruLlama(
    sentence_window_engine,
    app_id="App_1",
    feedbacks=[
        f_qa_relevance,
        f_qs_relevance,
        f_groundedness
    ]
)

for question in eval_questions:
    with tru_recorder as recording:
        sentence_window_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])
records.head()

pd.set_option("display.max_colwidth", None)
records[["input", "output"] + feedback]
tru.get_leaderboard(app_ids=[])
tru.run_dashboard()


## Window-sentence retrieval setup
# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
text = "hello. how are you? I am fine!  "
nodes = node_parser.get_nodes_from_documents([Document(text=text)])
print([x.text for x in nodes])
print(nodes[1].metadata["window"])
text = "hello. foo bar. cat dog. mouse"
nodes = node_parser.get_nodes_from_documents([Document(text=text)])
print([x.text for x in nodes])
print(nodes[0].metadata["window"])


### Building the index

from llama_index.core import Settings
Settings.llm=llm
Settings.embed_model="local:BAAI/bge-small-en-v1.5"
Settings.node_parser=node_parser
sentence_index = VectorStoreIndex.from_documents(
    [document], settings=Settings
)
sentence_index.storage_context.persist(persist_dir="./sentence_index")


node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)
nodes = node_parser.get_nodes_from_documents([document])
from llama_index.core.node_parser import get_leaf_nodes
leaf_nodes = get_leaf_nodes(nodes)

nodes_by_id = {node.node_id: node for node in nodes}
parent_node = nodes_by_id[leaf_nodes[30].parent_node.node_id]
print(parent_node.text)
from llama_index.core import Settings
Settings.llm=llm
Settings.embed_model="local:BAAI/bge-small-en-v1.5"
Settings.node_parser=node_parser
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

automerging_index = VectorStoreIndex(
    leaf_nodes, storage_context=storage_context,
    # service_context=auto_merging_context
    settings=Settings
)
automerging_index.storage_context.persist(persist_dir="./merging_index")

if not os.path.exists("./sentence_index"):
    sentence_index = VectorStoreIndex.from_documents(
        [document],
        # service_context=sentence_context
        settings=Settings
    )
    sentence_index.storage_context.persist(persist_dir="./sentence_index")
else:
    sentence_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./sentence_index"),
        # service_context=sentence_context
        settings=Settings
    )

if not os.path.exists("./merging_index"):
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)
    automerging_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            # service_context=auto_merging_context
            settings=Settings
        )
    automerging_index.storage_context.persist(persist_dir="./merging_index")
else:
    automerging_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./merging_index"),
        # service_context=auto_merging_context
        settings=Settings
    )


### Building the postprocessor
postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)
scored_nodes = [NodeWithScore(node=x, score=1.0) for x in nodes]
nodes_old = [deepcopy(n) for n in nodes]
nodes_old[1].text
replaced_nodes = postproc.postprocess_nodes(scored_nodes)
print(replaced_nodes[1].text)

automerging_retriever = automerging_index.as_retriever(
    similarity_top_k=12
)
retriever = AutoMergingRetriever(
    automerging_retriever, 
    automerging_index.storage_context, 
    verbose=True
)


### Adding a reranker
# BAAI/bge-reranker-base
# link: https://huggingface.co/BAAI/bge-reranker-base
rerank = SentenceTransformerRerank(
    top_n=2, model="BAAI/bge-reranker-base"
)
query = QueryBundle("I want a dog.")
scored_nodes = [
    NodeWithScore(node=TextNode(text="This is a cat"), score=0.6),
    NodeWithScore(node=TextNode(text="This is a dog"), score=0.4),
]
reranked_nodes = rerank.postprocess_nodes(
    scored_nodes, query_bundle=query
)
print([(x.text, x.score) for x in reranked_nodes])


rerank = SentenceTransformerRerank(top_n=6, model="BAAI/bge-reranker-base")
auto_merging_engine = RetrieverQueryEngine.from_args(
    automerging_retriever, node_postprocessors=[rerank]
)
auto_merging_response = auto_merging_engine.query(
    "Why should one use unsupervised learning for exoplanet detection?"
)
print(str(auto_merging_response))


### Runing the query engine
sentence_window_engine = sentence_index.as_query_engine(
    similarity_top_k=6, node_postprocessors=[postproc, rerank]
)
window_response = sentence_window_engine.query(
    "Why should one use unsupervised learning for exoplanet detection?"
)
print(str(window_response))


### Two layers
auto_merging_index_0 = build_automerging_index(
    documents,
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index_0",
    chunk_sizes=[2048,512],
)
auto_merging_engine_0 = get_automerging_query_engine(
    auto_merging_index_0,
    similarity_top_k=12,
    rerank_top_n=6,
)
tru_recorder = get_prebuilt_trulens_recorder(
    auto_merging_engine_0,
    app_id ='app_0'
)

def run_evals(eval_questions, tru_recorder, query_engine):
    for question in eval_questions:
        with tru_recorder as recording:
            response = query_engine.query(question)
    
    return tru_recorder

tru_recorder = run_evals(eval_questions, tru_recorder, auto_merging_engine_0)
Tru().get_leaderboard(app_ids=[])
Tru().run_dashboard()


### Three layers
auto_merging_index_1 = build_automerging_index(
    documents,
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index_1",
    chunk_sizes=[2048,512,128],
)
auto_merging_engine_1 = get_automerging_query_engine(
    auto_merging_index_1,
    similarity_top_k=12,
    rerank_top_n=6,
)
tru_recorder = get_prebuilt_trulens_recorder(
    auto_merging_engine_1,
    app_id ='app_1'
)
tru_recorder = run_evals(eval_questions, tru_recorder, auto_merging_engine_1)
Tru().get_leaderboard(app_ids=[])
Tru().run_dashboard()
