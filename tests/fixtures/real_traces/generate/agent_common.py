"""A tiny offline agent (LangGraph ReAct, a scripted fake model, a retriever tool) for traces.

Each task makes several searches before answering, rewriting its query as it goes, as agents
do. The model is scripted so the searches are fixed; the retriever is the RAG fixture's Chroma
store. Task ``t3`` plants a bad rewrite at step 2 (it searches for billing while the task asks
about support). Not imported by the test suite: it needs LangGraph and the instrumentation
libraries listed in the README.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from rag_common import DOCS, _metadata

TASKS = [
    ("t1", "Reset my password and turn on two-factor", ["reset password", "two-factor setup"]),
    ("t3", "How fast does support reply?", ["support reply time", "billing invoices", "support"]),
]


class ScriptedModel(GenericFakeChatModel):
    """Plays a fixed list of messages; tool binding is accepted and ignored."""

    def bind_tools(self, tools, **kwargs):
        return self


def _script():
    for task, _, searches in TASKS:
        for n, query in enumerate(searches):
            yield AIMessage(
                content="",
                tool_calls=[{"name": "search_kb", "args": {"query": query}, "id": f"{task}-{n}"}],
            )
        yield AIMessage(content=f"Answer for {task}.")


def build():
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding

    store = Chroma(
        collection_name="support-kb", embedding_function=DeterministicFakeEmbedding(size=32)
    )
    store.add_documents(
        [Document(page_content=text, metadata=_metadata(d)) for d, text in DOCS],
        ids=[d for d, _ in DOCS],
    )
    retriever = store.as_retriever(search_kwargs={"k": 2})

    @tool
    def search_kb(query: str) -> str:
        """Search the support knowledge base."""
        docs = retriever.invoke(query)
        return "\n".join(f"[{d.metadata['doc_id']}] {d.page_content}" for d in docs)

    model = ScriptedModel(messages=_script())
    return create_react_agent(model, [search_kb])


def run(agent):
    for task, question, _ in TASKS:
        agent.invoke(
            {"messages": [("user", question)]},
            config={"metadata": {"task_id": task}, "run_name": f"task {task}"},
        )
