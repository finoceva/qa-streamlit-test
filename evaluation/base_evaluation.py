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
    """
    Creates a QA testset by generating question-context pairs from the given nodes using the specified language model.

    Args:
        nodes (list): The list of nodes to generate the question-context pairs from.
        llm (LanguageModel): The language model to use for generating the questions.
        eval_dataset_path (str, optional): The path to save the generated QA testset JSON file. Default is "data/eval_testset.json".
        num_questions_per_chunk (int, optional): The number of questions to generate per chunk of nodes. Default is 2.

    Returns:
        None
    """
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
    """
    Run evaluation on a given index and nodes dictionary.

    Args:
        index_dict (Dict[str, BaseRetriever]): A dictionary mapping index names to retriever objects.
        nodes_dict (Dict[str, Node]): A dictionary mapping node names to node objects.
        eval_dataset_path (str, optional): Path to the evaluation dataset. Defaults to "data/eval_testset_v0.json".
        top_k (int, optional): The number of retrievals to consider. Defaults to 5.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the evaluation results for each index.
    """
    results = []
    eval_dataset = EmbeddingQAFinetuneDataset.from_json(eval_dataset_path)
    for name, index in index_dict.items():
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
