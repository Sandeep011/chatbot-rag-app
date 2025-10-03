# v0

import re

def clean_text(text: str) -> str:
    """
    Replace NULL characters and tabs/spaces with a whitespace.
    Replace windows/mac line breaks with Unix standard line breaks.
    Ensures atmost 2 consecutive newlines

    args:
        text: text to be cleaned

    return:
        text_cleaned: cleaned text

    """
    text = text.replace("\x00"," ") # replace NULL character
    text = re.sub(r"[ \t]+", " ", text) # collapse tabls/spaces
    text = re.sub(r"\r\n?", "\n", text) # CRLF/LR -> LF
    text = re.sub(r"\n{3,}", "\n\n", text) # limit newlines to atmost 2
    text_cleaned = text.strip()
    return text_cleaned


def text_to_chunks(text: str, max_chars: int = 900, overlap: int = 150):
    """
    Split text into chunks of text with respect to the max_chars limit and overlap clause.

    args:
        text: input text to be broekn to chunks
        max_chars: max size of a chunk
        overlap: this ensure that current data chunk has context of previous data chunk,
            so that when a chunk is processed, it has the necessary information or context.
    
    return:
        chunks: text data broken into chunks, adhering to the max_chars and overlap bounds.
    """
    text = text.strip()
    text_size = len(text)

    if not text:
        return []
    
    if text_size <= max_chars:
        return text
    
    start = 0
    chunks = []
    while start < text_size:
        end = min(text_size, start + max_chars)
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end >= text_size:
            break
        start = max(0, end - overlap)
    
    return chunks


def page_to_chunks(page_text: str):
    """
    Convert the text form a page into chunks of text.

    args:
        page_text: text data of the page
    
    return:
        page_chunks: chunks of text data of the page
    """
    clean_page_text = clean_text(page_text)
    page_chunks = text_to_chunks(clean_page_text)
    return page_chunks
