from typing import Dict, List
from llama_index.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
from datasets import load_dataset
import pandas as pd
from llama_index.evaluation import RetrieverEvaluator, get_retrieval_results_df
from llama_index.retrievers import BaseRetriever, RecursiveRetriever
from querying.query import HybridRetriever, get_bm25_retriever, get_vector_retriever
from llama_index.data_structs import Node
import nest_asyncio

nest_asyncio.apply()


def create_qa_testset(
    nodes, llm, eval_dataset_path="data/eval_testset.json", num_questions_per_chunk=2
):
    qa_dataset = generate_question_context_pairs(
        nodes, llm=llm, num_questions_per_chunk=num_questions_per_chunk
    )
    qa_dataset.save_json(eval_dataset_path)


def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    columns = {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}

    metric_df = pd.DataFrame(columns)

    return metric_df


async def run_eval(
    index_dict: Dict[str, BaseRetriever],
    nodes_dict: Dict[str, Node],
    eval_dataset_path: str = "data/eval_testset_v0.json",
    top_k: int = 5,
):
    results = []
    eval_dataset = EmbeddingQAFinetuneDataset.from_json(eval_dataset_path)
    for name, index in index_dict.items():
        # if name == "Hybrid":
        #     bm25_retriever = get_bm25_retriever(nodes_dict, top_k)
        #     vector_retriever = get_vector_retriever(index, top_k)
        #     retriever = HybridRetriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)
        if name == "Small2Big":
            vector_index_chunk = index.as_retriever(similarity_top_k=top_k)
            retriever = RecursiveRetriever(
                "vector",
                retriever_dict={"vector": vector_index_chunk},
                node_dict=nodes_dict,
                verbose=True,
            )
        else:
            retriever = index.as_retriever(similarity_top_k=top_k)

        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=retriever
        )
        result = await retriever_evaluator.aevaluate_dataset(
            eval_dataset,
            show_progress=True,
        )
        results.append(result)
    return results


if __name__ == "__main__":
    results = run_eval(eval_dataset_path="data/eval_testset_v0.json")
    print(results)
