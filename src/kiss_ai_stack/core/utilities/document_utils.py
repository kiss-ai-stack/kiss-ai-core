import os
from typing import List, Dict, Tuple

import pandas as pd
import tiktoken
from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_text, elements_to_text


def file_to_docs(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[List[str], List[Dict]]:
    """
    Converts a file into chunks and metadata for vector database storage.

    Args:
        file_path (str): Path to the file.
        chunk_size (int): Size of each chunk in tokens.
        chunk_overlap (int): Overlap between chunks in tokens.

    Returns:
        Tuple[List[str], List[Dict]]:
        - First element: List of document text chunks
        - Second element: List of corresponding metadata dictionaries
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'DocUtils :: The file {file_path} does not exist.')

    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError('DocUtils :: Invalid chunk size or overlap. Chunk size must be positive, '
                         'and overlap must be non-negative and less than chunk size.')

    try:
        if file_path.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
            dfs = pd.read_excel(file_path, sheet_name=None)
            text_content = []
            for sheet_name, df in dfs.items():
                sheet_text = f"Sheet: {sheet_name}\n"
                sheet_text += df.to_string(index=False)
                text_content.append(sheet_text)
            text_content = '\n\n'.join(text_content)
        else:
            elements = partition(filename=file_path)
            text_parts = []
            for element in elements:
                if isinstance(element, list):
                    text_parts.append(convert_to_text(element))
                else:
                    text_parts.append(elements_to_text([element]))
            text_content = '\n'.join(text_parts)

    except Exception as e:
        raise ValueError(f'DocUtils :: Error parsing file {file_path}: {str(e)}')

    encoding = tiktoken.get_encoding('cl100k_base')
    tokens = encoding.encode(text_content)
    chunks = []
    metadata_list = []

    for start in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[start:start + chunk_size]
        chunk = encoding.decode(chunk_tokens)

        if chunk.strip():
            chunks.append(chunk)

            metadata_list.append({
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'start_token': start,
                'end_token': start + len(chunk_tokens),
                'total_tokens': len(chunk_tokens)
            })

    return chunks, metadata_list
