
import ast
from types import SimpleNamespace

from typing import Any, Optional

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai_messages_token_helper import get_token_limit

from approaches.approach import Approach, ThoughtStep
from approaches.promptmanager import PromptManager
from core.authentication import AuthenticationHelper

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.planners import SequentialPlanner
from semantic_kernel.kernel import Kernel
from approaches.skplugins.RAG import RAGPlugin


class SKPlanRetrievalApproach(Approach):
    """
    A multi-step approach that first uses OpenAI to turn the user's input into a plan, considering the chat history.
    then uses Azure AI Search to retrieve relevant documents, iteratively as needed,
    finally using the user input and search results, use OpenAI to generate a response.
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

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(q))

        kernel = Kernel()
        service_id = "skplanretrieval"
        ai_service = AzureChatCompletion(service_id=service_id)
        kernel.add_service(ai_service)
        kernel.add_plugin(parent_directory="./approaches/skplugins", plugin_name="KBPlugin")
        kernel.add_plugin(RAGPlugin(), plugin_name="RAGPlugin")
        arguments = KernelArguments(original_input=q)
        planner = SequentialPlanner(service_id=service_id, kernel=kernel)
        goal = f"""
            Answer the user's query: {q}
            Use the tools that you have available to you to provide the best possible answer.
            If multiple iterations are needed, do them before responding to the user.
            If the user's query is unclear, suggest a clarifying question as part of your answer.
        """
        plan = await planner.create_plan(goal=goal)
        result = await plan.invoke(kernel=kernel, arguments=arguments)

        # print(result)
        output_docs = f"[{arguments.get('DOCUMENTS')}]"
        raw_docs = ast.literal_eval(output_docs)
        docs = [SimpleNamespace(**doc) for doc in raw_docs]
        
        executed_steps = []
        for step in plan._steps:
            executed_steps.append(
                ThoughtStep(
                    step.description,
                    arguments.get(step._outputs[0]),
                )
            )

        
        # Process results
        text_sources = self.get_sources_content(docs, use_semantic_captions, use_image_citation=False)

        extra_info = {
            "data_points": {"text": text_sources},
            "thoughts": executed_steps,
        }

        return {
            "message": {
                "content": str(result),
                "role": "assistant",
            },
            "context": extra_info,
            "session_state": session_state,
        }
