from __future__ import annotations

import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Literal, TypedDict, cast

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.observability import langchain_config
from app.retrieval.rag import Department, answer_department, format_sources, get_llm


class RouteQuery(BaseModel):
    """Route a user question to the correct department assistant."""

    department: Literal["hr", "finance", "both"] = Field(
        ...,
        description=(
            "Which department should answer this question? "
            "'hr' for HR topics like PTO, benefits, remote work, performance reviews, onboarding, and conduct. "
            "'finance' for Finance topics like expenses, travel, budgets, AP, procurement, and revenue. "
            "'both' if the question requires information from both departments."
        ),
    )


class DepartmentQuestions(BaseModel):
    """Split a cross-department question into department-specific questions."""

    hr_question: str = Field(
        ...,
        description="The part of the user's question that HR should answer.",
    )
    finance_question: str = Field(
        ...,
        description="The part of the user's question that Finance should answer.",
    )


class AskResult(TypedDict):
    answer: str
    sources: list[dict[str, str | None]]
    department_routed: Literal["hr", "finance", "both"]


ROUTER_SYSTEM = """You are a router that decides which department's assistant should answer a user's question.

- HR covers: PTO, parental leave, remote work, performance reviews, onboarding, code of conduct, learning and development, health and wellness benefits.
- Finance covers: expense reimbursement, corporate travel costs, budgets, accounts payable, revenue recognition, corporate cards, financial reporting, procurement.

If a question requires information from BOTH departments, return 'both'."""

SPLIT_SYSTEM = """You split cross-department employee policy questions into two focused questions.

- HR should receive only HR topics such as PTO, leave, benefits, remote work, onboarding, performance, wellness, and conduct.
- Finance should receive only Finance topics such as expenses, travel budgets, reimbursement, procurement, accounts payable, cards, and reporting.

Keep each split question concise and answerable by that department's policy documents.

Return only JSON with this exact shape:
{{"hr_question":"...","finance_question":"..."}}"""

HR_TERMS = {
    "benefit",
    "conduct",
    "harassment",
    "health",
    "hr",
    "leave",
    "onboarding",
    "parental",
    "performance",
    "pto",
    "remote",
    "vacation",
    "wellness",
}

FINANCE_TERMS = {
    "ap",
    "budget",
    "card",
    "expense",
    "finance",
    "hotel",
    "invoice",
    "payment",
    "per diem",
    "procurement",
    "purchase",
    "reimbursement",
    "travel",
}

VAGUE_PATTERNS = (
    "how much do i get",
    "what is the deadline",
    "do i need approval",
    "what happens if i submit it late",
)


def _contains_any(text: str, terms: set[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def is_vague_subquestion(question: str) -> bool:
    normalized = question.lower().strip(" ?.!")
    if normalized in VAGUE_PATTERNS:
        return True
    return not _contains_any(question, HR_TERMS) and not _contains_any(question, FINANCE_TERMS)


def split_user_questions(question: str) -> list[str]:
    normalized = " ".join(question.split())
    numbered_parts = [
        part.strip(" .")
        for part in re.split(r"(?:^|\s)\d+[.)]\s+", normalized)
        if part.strip(" .")
    ]
    if len(numbered_parts) > 1:
        return numbered_parts

    chunks = [
        chunk.strip()
        for chunk in re.findall(r"[^?]+\?", question)
        if chunk.strip(" .")
    ]
    if len(chunks) <= 1:
        return []
    return chunks


def route_question_fast(question: str) -> Literal["hr", "finance", "both"] | None:
    has_hr = _contains_any(question, HR_TERMS)
    has_finance = _contains_any(question, FINANCE_TERMS)

    if has_hr and has_finance:
        return "both"
    if has_hr:
        return "hr"
    if has_finance:
        return "finance"
    return None


def route_question(question: str) -> Literal["hr", "finance", "both"]:
    fast_route = route_question_fast(question)
    if fast_route is not None:
        return fast_route

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM),
            ("human", "{question}"),
        ]
    )
    structured_llm = get_llm().with_structured_output(RouteQuery)
    route = cast(
        RouteQuery,
        (prompt | structured_llm).invoke(
            {"question": question},
            config=langchain_config("department_router"),
        ),
    )
    return route.department


def split_department_questions_fast(question: str) -> DepartmentQuestions | None:
    chunks = [
        chunk.strip()
        for chunk in re.findall(r"[^?.!]+[?.!]?", question)
        if chunk.strip()
    ]
    if len(chunks) < 2:
        return None

    hr_chunks: list[str] = []
    finance_chunks: list[str] = []

    for chunk in chunks:
        has_hr = _contains_any(chunk, HR_TERMS)
        has_finance = _contains_any(chunk, FINANCE_TERMS)
        if has_hr and not has_finance:
            hr_chunks.append(chunk)
        elif has_finance and not has_hr:
            finance_chunks.append(chunk)

    if hr_chunks and finance_chunks:
        return DepartmentQuestions(
            hr_question=" ".join(hr_chunks),
            finance_question=" ".join(finance_chunks),
        )
    return None


def split_department_questions(question: str) -> DepartmentQuestions:
    fast_split = split_department_questions_fast(question)
    if fast_split is not None:
        return fast_split

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SPLIT_SYSTEM),
            ("human", "{question}"),
        ]
    )
    output = (prompt | get_llm() | StrOutputParser()).invoke(
        {"question": question},
        config=langchain_config("department_question_split"),
    )
    try:
        parsed = json.loads(output)
        return DepartmentQuestions.model_validate(parsed)
    except (json.JSONDecodeError, ValueError):
        return DepartmentQuestions(
            hr_question=question,
            finance_question=question,
        )


def _answer_single_question(question: str, department: Department | None = None) -> AskResult:
    if department is not None:
        answer, docs = answer_department(question, department)
        return {
            "answer": answer,
            "sources": format_sources(docs),
            "department_routed": department,
        }

    route = route_question(question)
    if route in ("hr", "finance"):
        answer, docs = answer_department(question, route)
        return {
            "answer": answer,
            "sources": format_sources(docs),
            "department_routed": route,
        }

    split_questions = split_department_questions(question)
    with ThreadPoolExecutor(max_workers=2) as executor:
        hr_future = executor.submit(answer_department, split_questions.hr_question, "hr")
        finance_future = executor.submit(
            answer_department,
            split_questions.finance_question,
            "finance",
        )
        hr_answer, hr_docs = hr_future.result()
        finance_answer, finance_docs = finance_future.result()

    answer = "HR:\n" f"{hr_answer}\n\n" "Finance:\n" f"{finance_answer}"
    return {
        "answer": answer,
        "sources": format_sources([*hr_docs, *finance_docs]),
        "department_routed": "both",
    }


def answer_question(question: str, department: Department | None = None) -> AskResult:
    subquestions = split_user_questions(question) if department is None else []
    if len(subquestions) > 1:
        answers: list[str] = []
        sources: list[dict[str, str | None]] = []
        routed_departments: set[Literal["hr", "finance", "both"]] = set()

        with ThreadPoolExecutor(max_workers=min(len(subquestions), 4)) as executor:
            futures: list[Future[AskResult] | None] = []
            for subquestion in subquestions:
                if is_vague_subquestion(subquestion):
                    futures.append(None)
                else:
                    futures.append(executor.submit(_answer_single_question, subquestion))
            for index, (subquestion, future) in enumerate(zip(subquestions, futures, strict=True), start=1):
                if future is None:
                    answers.append(
                        f"{index}. {subquestion}\nPlease clarify what this question refers to, or ask it with the policy topic included."
                    )
                    continue

                result = future.result()
                answers.append(f"{index}. {subquestion}\n{result['answer']}")
                sources.extend(result["sources"])
                routed_departments.add(result["department_routed"])

        if routed_departments == {"hr"}:
            routed: Literal["hr", "finance", "both"] = "hr"
        elif routed_departments == {"finance"}:
            routed = "finance"
        else:
            routed = "both"

        return {
            "answer": "\n\n".join(answers),
            "sources": sources,
            "department_routed": routed,
        }

    return _answer_single_question(question, department)
