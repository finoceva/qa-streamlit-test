import re
import camelot
import pandas as pd
from typing import List
from pathlib import Path


def get_tables(path: str, pages: List[int]):
    table_dfs = []
    for page in pages:
        table_list = camelot.read_pdf(path, pages=str(page))
        if len(table_list) == 0:
            table_dfs.append(pd.DataFrame())
            continue
        table_df = table_list[0].df
        table_df = (
            table_df.rename(columns=table_df.iloc[0])
            .drop(table_df.index[0])
            .reset_index(drop=True)
        )
        table_dfs.append(table_df)
    return table_dfs


def dataframe_to_text(df):
    if df.shape[0] == 0:
        return ""
    table_text = "Table:\n"
    columns = df.columns.tolist()
    for index, row in df.iterrows():
        row_text = ""
        for col in columns:
            row_text += f"{col}: {row[col]}\n"
        table_text += row_text + ";\n"

    return re.sub("(cid:1)", "Yes", table_text)


def extract_text_after_word(text, word):
    index = text.find(word)
    if index == -1:
        return text
    desired_output = text[index + len(word) :].strip()
    return desired_output
