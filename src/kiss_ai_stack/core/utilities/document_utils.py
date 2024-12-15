import asyncio
import os
from typing import List, Dict, Tuple

import pandas as pd
import tiktoken
from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_text, elements_to_text


async def file_to_docs(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[
    List[str], List[Dict[str, str]]]:
    """
    Convert a file into token-based chunks and associated metadata asynchronously.

    :param file_path: The absolute path to the input file to be processed.
    :param chunk_size: The maximum size of each chunk in tokens. Defaults to 1000.
    :param chunk_overlap: The overlap size between consecutive chunks in tokens. Defaults to 200.

    :returns:
        A tuple containing:
        - A list of document text chunks split based on the specified chunk size and overlap.
        - A list of metadata dictionaries corresponding to each text chunk.

    :raises FileNotFoundError: If the file does not exist at the specified path.
    :raises ValueError: If chunk parameters are invalid or if the file cannot be parsed.

    :example:
        file_path = "example.xlsx"
        chunks, metadata = await async_file_to_docs(file_path, chunk_size=500, chunk_overlap=100)
        print(chunks[0])  # Displays the first chunk
        print(metadata[0])  # Displays metadata for the first chunk
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DocUtils :: File not found: {file_path}")

    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "DocUtils :: Invalid chunk size or overlap. "
            "Chunk size must be positive, and overlap must be non-negative and less than chunk size."
        )

    try:
        if file_path.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
            dfs = pd.read_excel(file_path, sheet_name=None)
            text_content = "\n\n".join(
                f"Sheet: {sheet_name}\n{df.to_string(index=False)}"
                for sheet_name, df in dfs.items()
            )
        else:
            elements = await asyncio.to_thread(partition, filename=file_path)
            text_content = "\n".join(
                convert_to_text(element) if isinstance(element, list)
                else elements_to_text([element])
                for element in elements
            )
    except Exception as e:
        raise ValueError(f"DocUtils :: Failed to parse file '{file_path}'. Error: {str(e)}")

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text_content)
    chunks, metadata_list = [], []
    step_size = chunk_size - chunk_overlap

    for start in range(0, len(tokens), step_size):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text)
            metadata_list.append({
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "start_token": start,
                "end_token": end,
                "total_tokens": len(chunk_tokens),
            })

    return chunks, metadata_list
