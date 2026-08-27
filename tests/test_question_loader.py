import json

import pytest

from planner import QuestionLoadError, load_question, load_question_dict, load_questions


def test_load_question_dict_minimal():
    q = load_question_dict(
        {
            "question_id": "astro_001",
            "question": "Is the reported low water abundance real or a retrieval artifact?",
        }
    )
    assert q.question_id == "astro_001"
    assert "water abundance" in q.question
    assert q.domain == "generic"


def test_extra_keys_preserved_in_metadata():
    q = load_question_dict(
        {"question_id": "q1", "question": "?", "source_repo": "paper3"}
    )
    assert q.metadata["source_repo"] == "paper3"


def test_missing_required_key_raises():
    with pytest.raises(QuestionLoadError):
        load_question_dict({"question": "no id"})
    with pytest.raises(QuestionLoadError):
        load_question_dict({"question_id": "q1"})


def test_load_question_from_file(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"question_id": "q1", "question": "Why?"}))
    q = load_question(path)
    assert q.question_id == "q1"


def test_load_questions_list_and_jsonl(tmp_path):
    rows = [
        {"question_id": "q1", "question": "A?"},
        {"question_id": "q2", "question": "B?"},
    ]
    list_path = tmp_path / "qs.json"
    list_path.write_text(json.dumps(rows))
    assert [q.question_id for q in load_questions(list_path)] == ["q1", "q2"]

    jsonl_path = tmp_path / "qs.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows))
    assert [q.question_id for q in load_questions(jsonl_path)] == ["q1", "q2"]
