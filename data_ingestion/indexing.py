from dotenv import load_dotenv

load_dotenv()
import os
import pinecone
import chromadb
import qdrant_client
from llama_index import ServiceContext
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms import OpenAI, Anyscale
from llama_index.data_structs import Node
from misc.configuration import (
    DEFAULT_INDEX_NAME,
    EMBEDDING_DIMENSIONS,
    MAX_CONTEXT_LENGTHS,
)
from typing import List, Any, Literal


class Indexing:
    def __init__(
        self,
        llm: str,
        embed_model_name: str,
        embed_model: Any,
        vector_db: Literal["Base", "Pinecone", "Qdrant", "Chroma"] = "Pinecone",
        index_name: str = DEFAULT_INDEX_NAME,
    ):
        self.vector_db = vector_db
        self.index_name = index_name
        self.llm = llm
        self.embed_model_name = embed_model_name
        self.embed_model = embed_model
        self.embed_model_dimension = EMBEDDING_DIMENSIONS.get(self.embed_model_name)
        self.llm_max_tokens = MAX_CONTEXT_LENGTHS.get(self.llm, 4096)

        if "gpt-" in self.llm:
            llm_model = OpenAI(
                model=self.llm,
                temperature=0.0,
                max_tokens=min(1500, self.llm_max_tokens),
            )
            self.service_context = ServiceContext.from_defaults(
                llm=llm_model, embed_model=embed_model
            )
        else:
            try:
                llm_model = Anyscale(
                    model=self.llm, temperature=0.0, max_tokens=self.llm_max_tokens
                )
                self.service_context = ServiceContext.from_defaults(
                    llm=llm_model, embed_model=embed_model
                )
            except:
                raise Exception("Only support llm models from OpenAI or Anyscale APIs")

        if self.vector_db == "Pinecone":
            # load credentials
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

            pinecone.create_index(
                self.index_name,
                dimension=self.embed_model_dimension,
                metric="euclidean",
                pod_type="p1",
            )
            self.pinecone_index = pinecone.Index(self.index_name)

        elif self.vector_db == "Qdrant":
            # load credentials
            self.qdrant_client = qdrant_client.QdrantClient(
                url=os.getenv("QDRANT_URI"),
                api_key=os.getenv("QDRANT_API_KEY"),
                #prefer_grpc=True,
            )

    def indexing_and_storing(self, nodes: List[Node]):
        if self.vector_db == "Base":
            self.index_chunk = VectorStoreIndex(
                nodes, service_context=self.service_context
            )
            # saving index to disk for default options
            if not os.path.exists("tmp/default_vector_index"):
                os.makedirs("tmp/default_vector_index")
                self.index_chunk.storage_context.persist(
                    persist_dir="/tmp/default_vector_index"
                )

        elif self.vector_db == "Pinecone":
            vector_store = PineconeVectorStore(
                pinecone_index=self.pinecone_index,
                add_sparse_vector=True,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            self.index_chunk = VectorStoreIndex(
                nodes,
                service_context=self.service_context,
                storage_context=storage_context,
            )

        elif self.vector_db == "Chroma":
            # initialize client, setting path to save data
            db = chromadb.PersistentClient(path="./chroma_db")

            # create collection
            chroma_collection = db.get_or_create_collection(self.index_name)

            # assign chroma as the vector_store to the context
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # create your index
            self.index_chunk = VectorStoreIndex(
                nodes,
                service_context=self.service_context,
                storage_context=storage_context,
            )
            # saving index to disk for chroma option
            # if not os.path.exists("tmp/chroma_vector_index"):
            #     os.makedirs("tmp/chroma_vector_index")
            #     self.index_chunk.storage_context.persist(
            #         persist_dir="/tmp/chroma_vector_index"
            #     )

        elif self.vector_db == "Qdrant":
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.index_name,
                #prefer_grpc=True,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # create index
            self.index_chunk = VectorStoreIndex(
                nodes, service_context=self.service_context, storage_context=storage_context
            )
