import os
from llama_index.retrievers import (
    BaseRetriever,
    BM25Retriever,
    RecursiveRetriever,
)
from llama_index.indices.vector_store.retrievers.retriever import (
    VectorIndexRetriever,
)
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
)
from typing import List
from llama_index.data_structs import Node
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index import QueryBundle
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import ServiceContext
from llama_index.postprocessor.cohere_rerank import CohereRerank
from misc.configuration import COHERE_API_KEY


class Small2BigRetriever:
    def __init__(
        self,
        index: VectorStoreIndex,
        service_context: ServiceContext,
        nodes: List[Node],
        name: str = "Base",
        top_k: int = 5,
        use_cohere: bool = False,
    ):
        """
        Initializes the Base class.

        Args:
            index (VectorStoreIndex): The vector store index.
            service_context (ServiceContext): The service context.
            nodes (List[Node]): The list of nodes.
            name (str, optional): The name of the class. Defaults to "Base".
            top_k (int, optional): The number of top results to retrieve. Defaults to 5.
            use_cohere (bool, optional): Whether to use CohereRerank. Defaults to False.
        """
        self.name = name
        self.index = index
        self.service_context = service_context
        self.nodes = nodes
        self.top_k = top_k
        if name == "Small2Big":
            vector_index_chunk = index.as_retriever(similarity_top_k=top_k)
            nodes_dict = {n.node_id: n for n in nodes}
            self.retriever = RecursiveRetriever(
                "vector",
                retriever_dict={"vector": vector_index_chunk},
                node_dict=nodes_dict,
                verbose=True,
            )
        else:
            self.retriever = index.as_retriever(similarity_top_k=top_k)
        self.use_cohere = use_cohere
        if self.use_cohere:
            self.cohere_rerank = CohereRerank(
                api_key=COHERE_API_KEY, top_n=2, model="rerank-english-v2.0"
            )

    # TODO: add other options of node postprocessor: Filter low similarity thresholds nodes etc

    def get_response(self, query: str):
        query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            service_context=self.service_context,
            node_postprocessors=([self.cohere_rerank] if self.use_cohere else []),
        )
        response = query_engine.query(query)
        return response.response


def get_bm25_retriever(nodes: List[Node], top_k: int = 5):
    return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)


def get_vector_retriever(index: VectorStoreIndex, top_k: int = 5):
    return VectorIndexRetriever(index, similarity_top_k=top_k)


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

    def check_reranking(
        self,
        query: str,
        reranker: SentenceTransformerRerank,
    ):
        nodes = self.retrieve(query)
        reranked_nodes = reranker.postprocess_nodes(
            nodes,
            query_bundle=QueryBundle(query),
        )

        print("Initial retrieval: ", len(nodes), " nodes")
        print("Re-ranked retrieval: ", len(reranked_nodes), " nodes")


def get_response_from_hybrid_retriever(
    query: str,
    hybrid_retriever: HybridRetriever,
    service_context: ServiceContext,
    reranker: SentenceTransformerRerank,
):
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
        service_context=service_context,
    )
    response = query_engine.query(query)
    return str(response)
