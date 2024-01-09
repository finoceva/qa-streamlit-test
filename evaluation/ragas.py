from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from datasets import load_dataset, Dataset
from ragas.llama_index import evaluate
from langchain.chat_models import ChatOpenAI
from ragas.llms import LangchainLLM
from llama_index.query_engine import RetrieverQueryEngine
import pandas as pd

gpt4_turbo = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)

gpt4_turbo_wrapper = LangchainLLM(llm=gpt4_turbo)

# Use GPT 4 turbo to evaluate
faithfulness.llm = gpt4_turbo_wrapper
context_precision.llm = gpt4_turbo_wrapper
answer_relevancy.llm = gpt4_turbo_wrapper

# define metrics to evaluate (no need ground - truths)
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
]


# RAGAS Evaluate wrapper function for our case
def ragas_evaluate(
    query_engine: RetrieverQueryEngine,
    eval_dataset_path: str = "eval_data/eval_testset_v0.json",
):
    """
    Evaluates the given query engine on a test dataset.

    Args:
        query_engine (RetrieverQueryEngine): The query engine to be evaluated.
        eval_dataset_path (str, optional): The path to the evaluation dataset. Defaults to "eval_data/eval_testset_v0.json".

    Returns:
        pandas.DataFrame: The evaluation results, including the question, contexts, answer, context_precision, faithfulness, and answer_relevancy.
    """
    # convert eval dataset to ragas format
    eval_dataset = load_dataset("json", data_files=eval_dataset_path)["train"][0]
    questions = []
    contexts = []
    for hash_id in eval_dataset["queries"]:
        question = eval_dataset["queries"][hash_id]
        context_node_id = eval_dataset["relevant_docs"][hash_id][0]
        context = eval_dataset["corpus"][context_node_id]
        questions.append(question)
        contexts.append(context)
    eval_df = pd.DataFrame({"question": questions, "contexts": contexts})
    eval_df["answer"] = eval_df.apply(
        lambda x: query_engine.query(x["question"]).response, axis=1
    )
    eval_df["contexts"] = eval_df["contexts"].apply(lambda x: [x])
    result = evaluate(
        Dataset.from_pandas(eval_df),
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
        ],
    )
    return result.to_pandas()[
        [
            "question",
            "contexts",
            "answer",
            "context_precision",
            "faithfulness",
            "answer_relevancy",
        ]
    ]
