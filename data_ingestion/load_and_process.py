import re
import time
from llama_index import download_loader, Document
from misc.configuration import (
    SYSTEM_PROMPT_TEXT_PROCESSING,
    USER_PROMPT_TEXT_PROCESSING,
)
from misc.llm_gen import llm_gen, num_tokens_from_messages
from misc.text_utils import dataframe_to_text, get_tables, extract_text_after_word
from tqdm import tqdm


class PDFProcessor:
    def __init__(self, file):
        self.file = file

    def load_data(self):
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        self.documents = loader.load_data(file=self.file)

    def get_max_tokens_from_documents(self):
        self.max_tokens = max(
            [
                num_tokens_from_messages([{"role": "user", "content": doc.text}])
                for doc in self.documents
            ]
        )

    def get_tables_from_pdf(self):
        num_pages = len(self.documents)
        self.tables = get_tables(str(self.file), range(num_pages))

    def text_processing(self, timeout=10):
        new_documents = []
        for i, document in tqdm(enumerate(self.documents)):
            text_doc = re.sub(r"\n{3,}", "\n\n", document.text)
            table_data = self.tables[i]
            if table_data.shape[0] > 0:
                table_text = dataframe_to_text(table_data)
                params = {
                    "user_prompt": USER_PROMPT_TEXT_PROCESSING.format(
                        first_text=document.text, second_text=table_text
                    ),
                    "system_prompt": SYSTEM_PROMPT_TEXT_PROCESSING,
                    "MODEL": "gpt-3.5-turbo",
                    "max_tokens": max(
                        1200, self.max_tokens
                    ),  # upper bound = 1200 due to personal limits
                    "temperature": 0.0,
                }
                try:
                    text_doc = llm_gen(**params)
                    text_doc = extract_text_after_word(text_doc, "DESIRED OUTPUT:")
                except:
                    pass
            new_dict = {k: v for k, v in document.to_dict().items() if k != "text"}
            new_documents.append(Document(**new_dict, text=text_doc))
            if ((i + 1) % 20) == 0:
                time.sleep(timeout)
        self.new_documents = new_documents
