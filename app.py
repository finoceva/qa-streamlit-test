import streamlit as st
import pickle
import os
import pinecone
import chromadb
from pathlib import Path
from typing import Literal, Any
from data_ingestion.indexing import Indexing
from data_ingestion.embedding import get_embedding_model
from data_ingestion.load_and_process import PDFProcessor
from data_ingestion.node_parsing import (
    simple_convert_documents_into_nodes,
    small_to_big_conversion,
)
from querying.query import Small2BigRetriever
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore, ChromaVectorStore
from llama_index.llms import OpenAI
from llama_index.llms.anyscale import Anyscale
from misc.configuration import (
    ANYSCALE_API_KEY,
    DEFAULT_INDEX_NAME,
    EMBEDDING_MODEL_NAMES_TO_INDEX_NAMES,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    EMBEDDING_DIMENSIONS,
)



def get_retriever(
    api_key: str,
    llm_name: str,
    embed_model: Any,
    embedding_model_name: str,
    process_raw_documents: bool = False,
    use_cohere: bool = False,
    free_tier_pinecone_acc: bool = True,
) -> Small2BigRetriever:
    if process_raw_documents:
        EFS_DIR = Path("data")
        # path to pdf dataset
        DOCS_DIR = Path(EFS_DIR, "manual.pdf")

        # load and preprocess pdf file
        loader = PDFProcessor(str(DOCS_DIR))
        loader.load_data()
        loader.get_tables_from_pdf()
        loader.get_max_tokens_from_documents()
        loader.text_processing()
        documents = loader.new_documents

        # save documents
        with open("data/cleaned_documents.pkl", "wb") as f:
            pickle.dump(documents, f)
    else:
        with open("data/cleaned_documents.pkl", "rb") as f:
            documents = pickle.load(f)

    # create base nodes first
    base_nodes = simple_convert_documents_into_nodes(
        documents, chunk_size=512, chunk_overlap=20, re_name_node_ids=True
    )
    # create sub nodes
    all_nodes = small_to_big_conversion(base_nodes, sub_chunk_sizes=[256])

    if "gpt-" in llm_name:
        llm_model = OpenAI(
            model=llm_name, temperature=0.0, max_tokens=1500, api_key=api_key
        )
    else:
        llm_model = Anyscale(
            model=llm_name, temperature=0.0, max_tokens=4096, api_key=ANYSCALE_API_KEY
        )

    service_context = ServiceContext.from_defaults(
            llm=llm_model, embed_model=embed_model
        )
    index_name = EMBEDDING_MODEL_NAMES_TO_INDEX_NAMES.get(embedding_model_name)
    if not index_name:
        index_name = embedding_model_name.split("/")[-1] + "_" + DEFAULT_INDEX_NAME

    
    if EMBEDDING_DIMENSIONS.get(embedding_model_name) > 384:
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT"),
        )

        # Check if the indexes exist or create them if they don't
        existing_indexes = pinecone.list_indexes()

        if index_name not in existing_indexes:
            print("The index does not exist.")
            if free_tier_pinecone_acc and len(existing_indexes) > 0:
                # delete index first
                pinecone.delete_index(name=existing_indexes[0], timeout=-1)
                print("Deleted existing index")

            # create Index and Storing
            indexing = Indexing(
                llm=llm_name,
                embed_model_name=embedding_model_name,
                embed_model=embed_model,
                vector_db="Pinecone",
                index_name=index_name,
            )
            indexing.indexing_and_storing(all_nodes)
        vector_store = PineconeVectorStore(pinecone_index=pinecone.Index(index_name))
    else:
        if not os.path.exists("./chroma_db"):
            # create Index and Storing
            indexing = Indexing(
                llm=llm_name,
                embed_model_name=embedding_model_name,
                embed_model=embed_model,
                vector_db="Chroma",
                index_name=index_name,
            )
            indexing.indexing_and_storing(all_nodes)
        else:
            # load from disk
            db2 = chromadb.PersistentClient(path="./chroma_db")
            chroma_collection = db2.get_or_create_collection(index_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, service_context=service_context
    )

    return Small2BigRetriever(
        index,
        service_context,
        all_nodes,
        name="Small2Big",
        top_k=5,
        use_cohere=use_cohere,
    )


# Page title
st.set_page_config(page_title="🦜🔗 Question Answering Tool for RENNER Compressors 🦜🔗")
st.title("🦜🔗 Question Answering Tool for RENNER Compressors 🦜🔗")


@st.cache_resource
def cached_get_embedding_model(embedding_model_name):
    return get_embedding_model(embedding_model_name)


def run():
    # LLM Model selection
    llm_name = st.selectbox(
        "Select LLM Model",
        [
            # "gpt-3.5-turbo", # not enough credits
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.1",
            # "HuggingFaceH4/zephyr-7b-beta",
            # "Open-Orca/Mistral-7B-OpenOrca",
        ],  # Add more models as needed
    )

    # Show the appropriate API key input based on the selected llm_model
    # api_key = st.text_input("OpenAI API Key", type="password")
    # anyscale_key = ""
    # if "gpt-" not in llm_name:

    # Embedding model (cached)
    embedding_model_name = st.selectbox(
        "Select Retrieval Embedding Model",
        [
            "text-embedding-ada-002",
            "local:BAAI/bge-small-en",
        ],
    )
    if embedding_model_name.startswith("local:"):
        # Display a loading message
        st.info(
            "Loading the embedding model. It may take longer than expected."
        )

        # Remove the loading message
        st.empty()

    if embedding_model_name in ["text-embedding-ada-002"]:
        embed_model = get_embedding_model(
            embedding_model_name,
            api_kwargs={"api_key": OPENAI_API_KEY, "api_base": OPENAI_API_BASE},
        )
    else:
        embed_model = cached_get_embedding_model(embedding_model_name)

    # Option selection use_cohere rerank
    use_cohere = st.checkbox("Re-rank before answering", value=False)

    retriever = get_retriever(
        api_key=OPENAI_API_KEY,
        embed_model=embed_model,
        embedding_model_name=embedding_model_name,
        llm_name=llm_name,
        use_cohere=use_cohere,
        free_tier_pinecone_acc=True,
    )

    # Query text input
    query_text = st.text_input(
        "Enter your question:",
        placeholder="How should I lubricate again for the motor bearings of this compressor?",
    )

    # Form input and query
    result = []
    with st.form("myform", clear_on_submit=True):
        submitted = st.form_submit_button("Submit", disabled=not (query_text))
        if submitted:
            with st.spinner(
                "Thanks for your question and your patience... Please wait..."
            ):
                response = retriever.get_response(query_text)
                result.append(response)

    if len(result):
        st.info(response)


if __name__ == "__main__":
    run()
