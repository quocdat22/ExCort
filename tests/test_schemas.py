"""Tests for public API validation boundaries."""

import pytest
from pydantic import ValidationError

from excort.schemas import ChatRequest, ChatResponse


def test_chat_request_accepts_valid_values() -> None:
    request = ChatRequest(question="What is MLOps?", top_k=4)
    assert request.top_k == 4


@pytest.mark.parametrize("top_k", [0, 21])
def test_chat_request_rejects_out_of_range_top_k(top_k: int) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(question="question", top_k=top_k)


def test_chat_response_requires_source_list() -> None:
    response = ChatResponse(question="question", answer="answer", sources=[])
    assert response.model_dump()["sources"] == []
