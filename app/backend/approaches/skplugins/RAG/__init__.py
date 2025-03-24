import json
import os
from typing import TypedDict, Annotated
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from semantic_kernel.functions.kernel_function_decorator import kernel_function

class Doc(TypedDict):
    id: str
    content: str
    sourcepage: str
    sourcefile: str

class RAGPlugin:
    docs: list[Doc] =[
        {"abc1", "This is a test", "test.py", "1-100"},
        {"abc2", "Lorem ipsum", "foo.py", "1-100"},
        {"abc3", "Hello world", "bar.py", "1-100"},
    ]

    @kernel_function
    async def get_docs(self, search_data: str) -> list[Doc]:
        """Search for documents in a KB using provided search attributes via Azure AI Search SDK"""
        
        # validate that search_query is a valid json
        # try:
        #     search_json = json.loads(search_data)
        # except json.JSONDecodeError as e:
        #     print(f"Error: {e}")
        #     return []
        
        # get search query and file filter
        # search_query = search_json.get("search_query", "")
        # file_filter = search_json.get("file_filter", "")

        search_query = search_data
        file_filter = None

        #if search_query is wrapped in quotes, remove them
        if search_query.startswith('"') and search_query.endswith('"'):
            search_query = search_query[1:-1]

        # print(f"Search query: {search_query} || File filter: {file_filter}")

        # Example placeholder values
        ai_search_name = os.getenv("AZURE_SEARCH_SERVICE")
        endpoint = f"https://{ai_search_name}.search.windows.net"
        index_name = os.getenv("AZURE_SEARCH_INDEX")
        
        #Get Token
        credential = DefaultAzureCredential()

        search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential
        )

        results = search_client.search(
            top=10,
            search_text=search_query,
            filter=f"sourcefile eq '{file_filter}'" if file_filter else None
        )

        docs = []
        for hit in results:
            docs.append({
                "id": hit.get("id", ""),
                "content": hit.get("content", ""),
                "sourcepage": hit.get("sourcepage", ""),
                "sourcefile": hit.get("sourcefile", "")
            })

        return docs
    

class RAGPlugin:
    docs: list[Doc] =[
        {"abc1", "This is a test", "test.py", "1-100"},
        {"abc2", "Lorem ipsum", "foo.py", "1-100"},
        {"abc3", "Hello world", "bar.py", "1-100"},
    ]

    @kernel_function
    async def get_docs(self, search_data: str) -> list[Doc]:
        """Search for documents in a KB using provided search attributes via Azure AI Search SDK"""
        
        # validate that search_query is a valid json
        # try:
        #     search_json = json.loads(search_data)
        # except json.JSONDecodeError as e:
        #     print(f"Error: {e}")
        #     return []
        
        # get search query and file filter
        # search_query = search_json.get("search_query", "")
        # file_filter = search_json.get("file_filter", "")

        search_query = search_data
        file_filter = None

        #if search_query is wrapped in quotes, remove them
        if search_query.startswith('"') and search_query.endswith('"'):
            search_query = search_query[1:-1]

        # print(f"Search query: {search_query} || File filter: {file_filter}")

        # Example placeholder values
        ai_search_name = os.getenv("AZURE_SEARCH_SERVICE")
        endpoint = f"https://{ai_search_name}.search.windows.net"
        index_name = os.getenv("AZURE_SEARCH_INDEX")
        
        #Get Token
        credential = DefaultAzureCredential()

        search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential
        )

        results = search_client.search(
            top=10,
            search_text=search_query,
            query_type="semantic",
            semantic_configuration_name="default",
            query_answer="extractive",
            filter=f"sourcefile eq '{file_filter}'" if file_filter else None
        )

        docs = []
        for hit in results:
            docs.append({
                "id": hit.get("id", ""),
                "content": hit.get("content", ""),
                "sourcepage": hit.get("sourcepage", ""),
                "sourcefile": hit.get("sourcefile", "")
            })

        return docs