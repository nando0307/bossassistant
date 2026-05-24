from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Literal, cast

from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jVector
from langchain_neo4j.vectorstores.neo4j_vector import SearchType
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from sentence_transformers import CrossEncoder

from app.config import settings
from app.observability import langchain_config

Department = Literal["hr", "finance"]
RetrievalMode = Literal["fast", "deep"]

MULTI_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate four
different versions of the given user question to retrieve relevant documents from a vector
database. Provide these alternative questions separated by newlines.

Original question: {question}

Note: Only return a list of questions under format:
1. question 1
2. question 2
3. question 3
4. question 4
without any explanation"""

RAG_TEMPLATE = """You are an assistant for the {department} department. Answer the user's
question based ONLY on the context below. If the context doesn't contain the answer,
say you don't have that information in your policies - do not guess.

Be concise. Cite the policy document title when relevant.
Answer only the specific question asked. Do not list unrelated policy categories just because they appear in context.
If the user asks whether a specific item, service, or expense is allowed, answer yes only when that specific item or a clearly matching category appears in the context.
If the context only gives general reimbursement rules or unrelated examples, say you don't have that specific information in your policies.
Do not combine, add, multiply, or recalculate figures unless the user explicitly asks for a calculation.
If a policy gives annual and monthly accrual rates for the same benefit, treat them as two ways to describe the same accrual policy. Do not add annual and monthly PTO figures together.
If a policy gives different amounts under different conditions, include the conditions instead of assuming which one applies to the user.
If the user asks for a budget and the policy gives a maximum, cap, or limit, answer with that limit.

Context:
{context}

Question: {question}

Answer:"""

INDEX_CONFIG: dict[Department, dict[str, str]] = {
    "hr": {
        "department_name": "HR",
        "index_name": "hr_vector",
        "keyword_index_name": "hr_keyword",
        "node_label": "HRDocument",
    },
    "finance": {
        "department_name": "Finance",
        "index_name": "fin_vector",
        "keyword_index_name": "fin_keyword",
        "node_label": "FinDocument",
    },
}


@lru_cache(maxsize=1)
def get_llm() -> ChatNVIDIA:
    return ChatNVIDIA(
        model=settings.nvidia_chat_model,
        nvidia_api_key=settings.nvidia_api_key.get_secret_value(),
        temperature=0,
        max_completion_tokens=settings.nvidia_max_tokens,
    )


@lru_cache(maxsize=1)
def get_embedder() -> NVIDIAEmbeddings:
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key=settings.nvidia_api_key.get_secret_value(),
        truncate="END",
    )


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder("BAAI/bge-reranker-large", max_length=512)


@lru_cache(maxsize=2)
def get_vector_store(department: Department) -> Neo4jVector:
    config = INDEX_CONFIG[department]
    return Neo4jVector.from_existing_graph(
        embedding=get_embedder(),
        node_label=config["node_label"],
        embedding_node_property="embedding",
        text_node_properties=["text"],
        index_name=config["index_name"],
        keyword_index_name=config["keyword_index_name"],
        search_type=SearchType.HYBRID,
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )


def parse_queries(text: str) -> list[str]:
    pattern = re.compile(r"^\s*\d+\.\s+(.+)$")
    return [match.group(1).strip() for line in text.splitlines() if (match := pattern.match(line))]


def generate_queries(question: str) -> list[str]:
    chain = ChatPromptTemplate.from_template(MULTI_QUERY_TEMPLATE) | get_llm() | StrOutputParser()
    return parse_queries(
        chain.invoke(
            {"question": question},
            config=langchain_config("multi_query_generation"),
        )
    )


def reciprocal_rank_fusion(results: list[list[Document]], k: int = 60) -> list[Document]:
    fused_scores: dict[str, float] = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            fused_scores[doc_str] = fused_scores.get(doc_str, 0.0) + 1 / (rank + k)

    reranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [loads(doc_str) for doc_str, _ in reranked]


def rerank(query: str, docs: list[Document], top_k: int = 4, mode: RetrievalMode = "fast") -> list[Document]:
    if not docs:
        return []
    if mode == "fast" or not settings.enable_reranker:
        return docs[:top_k]

    pairs = [(query, clean_page_content(doc.page_content)) for doc in docs]
    scores = get_reranker().predict(cast(Any, pairs))
    scored = sorted(zip(docs, scores, strict=True), key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


def retrieve(
    question: str,
    department: Department,
    mode: RetrievalMode = "fast",
    retrieval_candidates: int = 10,
    final_k: int = 4,
) -> list[Document]:
    vector_store = get_vector_store(department)
    alt_queries = generate_queries(question) if mode == "deep" else []
    all_queries = [question, *alt_queries]
    results = [vector_store.similarity_search(query, k=retrieval_candidates) for query in all_queries]
    fused = reciprocal_rank_fusion(results)
    return rerank(question, fused[:retrieval_candidates], top_k=final_k, mode=mode)


def clean_page_content(page_content: str) -> str:
    return page_content.removeprefix("\ntext: ").strip()


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{doc.metadata.get('title', '?')}] {clean_page_content(doc.page_content)}" for doc in docs)


def format_sources(docs: list[Document]) -> list[dict[str, str | None]]:
    return [
        {
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
            "department": doc.metadata.get("department"),
            "preview": clean_page_content(doc.page_content)[:240],
        }
        for doc in docs
    ]


def answer_department(question: str, department: Department, mode: RetrievalMode = "fast") -> tuple[str, list[Document]]:
    docs = retrieve(question, department, mode=mode)
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    answer = (prompt | get_llm() | StrOutputParser()).invoke(
        {
            "department": INDEX_CONFIG[department]["department_name"],
            "context": format_docs(docs),
            "question": question,
        },
        config=langchain_config(
            "department_answer",
            {"department": department, "source_count": str(len(docs))},
        ),
    )
    return answer, docs
