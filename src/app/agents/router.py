from __future__ import annotations

from typing import Literal, TypedDict, cast

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

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


class AskResult(TypedDict):
    answer: str
    sources: list[dict[str, str | None]]
    department_routed: Literal["hr", "finance", "both"]


ROUTER_SYSTEM = """You are a router that decides which department's assistant should answer a user's question.

- HR covers: PTO, parental leave, remote work, performance reviews, onboarding, code of conduct, learning and development, health and wellness benefits.
- Finance covers: expense reimbursement, corporate travel costs, budgets, accounts payable, revenue recognition, corporate cards, financial reporting, procurement.

If a question requires information from BOTH departments, return 'both'."""

SYNTHESIS_TEMPLATE = """The user asked a question that spans HR and Finance. Synthesize the two department answers below into a single coherent response. If they contradict, note it.

User question: {question}

HR answer:
{hr}

Finance answer:
{finance}

Final synthesized answer:"""


def route_question(question: str) -> Literal["hr", "finance", "both"]:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_SYSTEM),
            ("human", "{question}"),
        ]
    )
    structured_llm = get_llm().with_structured_output(RouteQuery)
    route = cast(RouteQuery, (prompt | structured_llm).invoke({"question": question}))
    return route.department


def answer_question(question: str, department: Department | None = None) -> AskResult:
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

    hr_answer, hr_docs = answer_department(question, "hr")
    finance_answer, finance_docs = answer_department(question, "finance")
    prompt = ChatPromptTemplate.from_template(SYNTHESIS_TEMPLATE)
    answer = (prompt | get_llm() | StrOutputParser()).invoke(
        {
            "question": question,
            "hr": hr_answer,
            "finance": finance_answer,
        }
    )
    return {
        "answer": answer,
        "sources": format_sources([*hr_docs, *finance_docs]),
        "department_routed": "both",
    }
