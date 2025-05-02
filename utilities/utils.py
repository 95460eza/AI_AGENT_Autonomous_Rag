

#from llama_index.core.settings import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional


# REMOVE CHARACTERS THAT CANNOT BE EMBEEDED BECAUSE OF NOT BEING "utf-8" ENCODEABLE
def safe_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")




def get_doc_tools(
        file_path: str,
        name: str,
        pre_trained_model: str,
        llm,
        similarity_top_k: int
) -> str:
    """Get vector query and summary query tools from a document."""

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    for node in nodes:
        node.text = safe_text(node.text)
        
    # filter out nodes with None or empty text
    nodes = [n for n in nodes if n.text is not None and n.text.strip() != ""]

    # Use a local embedding model
    embed_model = HuggingFaceEmbedding(pre_trained_model)
    # Create the vector index with the local embedding model
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)

    def vector_query(
            query: str,
            page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over a given paper.

        Useful if you have specific questions over the paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.

        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.

        """

        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]

        query_engine = vector_index.as_query_engine(
        llm=llm,
        similarity_top_k = similarity_top_k,
        filters = MetadataFilters.from_dicts(
            metadata_dicts,
            condition=FilterCondition.OR
        )
        )
        response = query_engine.query(query)
        return response

    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        description=(f"Useful for retrieving specific context from the {name} paper."),
        fn=vector_query
    )

    
    #USING AN LLM, "SummaryIndex" STORES "ORGANIZED" TEXT IN THE FORM OF "HIERARCHICAL" SUMMARIES
    #Set the global/default settings:
    #Settings.llm = llm
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="simple_summarize",
        llm=llm,
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to {name}"
        ),
    )

    return vector_query_tool, summary_tool

