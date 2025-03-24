
import ast
from types import SimpleNamespace
from collections.abc import Awaitable, Callable

from typing import Any, Optional

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai_messages_token_helper import get_token_limit

from approaches.approach import Approach, ThoughtStep
from approaches.promptmanager import PromptManager
from core.authentication import AuthenticationHelper

from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.filters import AutoFunctionInvocationContext, FilterTypes
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.kernel import Kernel

from approaches.skplugins.RAG import RAGPlugin


class SKReasonedApproach(Approach):
    """
    A reasoned approach that using Semantic-Kernel, OpenAI o1 and Azure AI Search as a RAG Plugin
    to get the information needed to answer the user's question.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_model: str,
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
        prompt_manager: PromptManager,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model, self.ALLOW_NON_GPT_MODELS)
        self.prompt_manager = prompt_manager
        self.answer_prompt = self.prompt_manager.load_prompt("ask_answer_question.prompty")




    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        q = messages[-1]["content"]
        if not isinstance(q, str):
            raise ValueError("The most recent message content must be a string.")
        overrides = context.get("overrides", {})
        seed = overrides.get("seed", None)
        auth_claims = context.get("auth_claims", {})
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        filter = self.build_filter(overrides, auth_claims)
        
        #Define a global variable called docs
        query = q
        docs = []

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(q))

        kernel = Kernel()

        @kernel.filter(filter_type=FilterTypes.AUTO_FUNCTION_INVOCATION)
        async def sk_function_handler(
            context: AutoFunctionInvocationContext, next: Callable[[AutoFunctionInvocationContext], Awaitable[None]]
        ) -> None:
            nonlocal docs
            nonlocal query
            await next(context)
            docs = context.function_result
            query = context.arguments.get("search_data")

        service_id = "skreasoned"
        reasoning_effort = "medium"
        max_tokens = 5000
        system_message = f"""
Assistant helps developers with their coding questions.
Focus only on answering the user's query using the tools available.
Always include the source name in-line for each fact you use in the response using square brackers, for example [info1.txt].
Don't combine sources, list each source separately, for example [info1.txt][info2.pdf].
"""        
        
        chat_service = AzureChatCompletion(service_id=service_id, instruction_role="developer")
        request_settings = AzureChatPromptExecutionSettings(
            service_id=service_id,
            max_completion_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )
        
        kernel.add_plugin(RAGPlugin(), plugin_name="RAGPlugin")

        chat_history = ChatHistory(system_message=system_message)
        chat_history.add_user_message(q)

        response = await chat_service.get_chat_message_content(
            chat_history=chat_history,
            settings=request_settings,
            kernel=kernel,
        )

        output_docs = f"{docs.value}"
        raw_docs = ast.literal_eval(output_docs)
        clean_docs = [SimpleNamespace(**doc) for doc in raw_docs]
        
        #Get all the values of sourcepage in clean_docs separated by a new line
        sourcepages = "\n".join([doc.sourcepage for doc in clean_docs])
        
        # Process results
        text_sources = self.get_sources_content(clean_docs, use_semantic_captions, use_image_citation=False)
        executed_steps = [
            ThoughtStep(
                "Searched in KB",
                f"Search query: {query}.\n"
            ),
            ThoughtStep(
                "Retrieved 10 documents",
                sourcepages
            ),
            ThoughtStep(
                "o1 response",
                f"Answer: {response}.\n"
            ),

        ]

        extra_info = {
            "data_points": {"text": text_sources},
            "thoughts": executed_steps,
        }

        return {
            "message": {
                "content": str(response),
                "role": "assistant",
            },
            "context": extra_info,
            "session_state": session_state,
        }
