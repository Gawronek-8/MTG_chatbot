from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_default_splitter() -> RecursiveCharacterTextSplitter:
    """
    Returns a default RecursiveCharacterTextSplitter with predefined chunk size and overlap.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )