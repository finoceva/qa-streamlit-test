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
)

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
    """
    Returns an embedding model based on the specified `embedding_model_name`.

    Parameters:
        embedding_model_name (str): The name of the embedding model to use.
        model_kwargs (dict): Additional keyword arguments for the embedding model constructor. Default is `{"device": device}`.
        encode_kwargs (dict): Additional keyword arguments for the embedding model's `encode` method. Default is `{"device": device, "batch_size": (100 if device == torch.device("cuda") else 1)}`.
        api_kwargs (dict): Additional keyword arguments for the API client. Default is `{"api_key": OPENAI_API_KEY, "api_base": OPENAI_API_BASE}`.

    Returns:
        embedding_model: The embedding model based on the specified `embedding_model_name`.
    """
    if embedding_model_name == "text-embedding-ada-002":
        embedding_model = OpenAIEmbedding(
            model=embedding_model_name,
            openai_api_base=api_kwargs.get("api_base", OPENAI_API_BASE),
            openai_api_key=api_kwargs.get("api_key", OPENAI_API_KEY),
        )

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
        """
        Initializes the object with the given model name and batch size.

        Parameters:
            model_name (str): The name of the model to be used for embedding.
            batch_size (int, optional): The batch size for encoding the embeddings. Default is 100.

        Returns:
            None
        """
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
