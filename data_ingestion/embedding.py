import os
import torch
import openai
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import OpenAIEmbedding
from llama_index.data_structs import Node
from llama_index.embeddings import resolve_embed_model
from typing import Dict, List
from misc.configuration import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
)  # , ANYSCALE_API_BASE, ANYSCALE_API_KEY

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_embedding_model(
    embedding_model_name,
    model_kwargs={"device": device},
    encode_kwargs={
        "device": device,
        "batch_size": (100 if device == torch.device("cuda") else 1),
    },
    api_kwargs: Dict = {"api_key": OPENAI_API_KEY, "api_base": OPENAI_API_BASE},
):
    if embedding_model_name == "text-embedding-ada-002":
        embedding_model = OpenAIEmbedding(
            model=embedding_model_name,
            openai_api_base=api_kwargs.get("api_base", OPENAI_API_BASE),
            openai_api_key=api_kwargs.get("api_key", OPENAI_API_KEY),
        )
    # elif embedding_model_name in []:
    #     client = openai.OpenAI(
    #             base_url = ANYSCALE_API_BASE,
    #             api_key = ANYSCALE_API_KEY,
    #         )
    #     embedding_model = client.embeddings.create(
    #             model="thenlper/gte-large",
    #             input="Your text string goes here",
    #         )
    elif embedding_model_name in [
        "local:BAAI/bge-small-en",
        "local:BAAI/bge-base-en",
        "local:BAAI/bge-large-en",
    ]:
        embedding_model = resolve_embed_model(embedding_model_name)
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,  # also works with model_path
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    return embedding_model


# TODO: Embed each node using a local embedding model / or via anyscale api
class EmbedNodes:
    def __init__(self, model_name, batch_size=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = get_embedding_model(
            model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"device": self.device, "batch_size": batch_size},
        )

    def __call__(self, nodes: List[Node]) -> List[Node]:
        texts = [node.text for node in nodes]
        embeddings = self.embedding_model.embed_documents(texts)
        assert len(nodes) == len(embeddings)

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding
        return nodes
