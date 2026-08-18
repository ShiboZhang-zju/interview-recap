from __future__ import annotations

import re
from typing import Any

import yaml

from .config import project_path


QUESTION_PATTERNS = (
    r"[？?]",
    r"(什么|为什么|怎么|如何|哪些|区别|原理|介绍一下|说一下|讲一下|能否|可不可以|有没有|是否|请你|你会)",
)
FOLLOW_UP_CUES = ("那", "那么", "继续", "具体", "为什么", "如果", "刚才", "进一步", "还有", "追问")
NEW_TOPIC_CUES = ("下一个问题", "换一个问题", "再问一个", "接下来", "另外一个方向", "最后一个问题")
ACKNOWLEDGEMENTS = {"嗯", "好", "好的", "可以", "明白", "行", "对", "继续"}


def question_score(text: str) -> int:
    return sum(1 for pattern in QUESTION_PATTERNS if re.search(pattern, text))


def infer_roles(segments: list[dict[str, Any]], minimum_score: int = 1) -> dict[str, str]:
    speakers = sorted({segment["speaker"] for segment in segments if segment["speaker"] != "speaker_unknown"})
    if not speakers:
        return {}
    scores = {speaker: 0 for speaker in speakers}
    turns = {speaker: 0 for speaker in speakers}
    for segment in segments:
        speaker = segment["speaker"]
        if speaker in scores:
            scores[speaker] += question_score(segment["text"])
            turns[speaker] += 1
    interviewer = max(speakers, key=lambda speaker: (scores[speaker], -speakers.index(speaker)))
    if scores[interviewer] < minimum_score and len(speakers) > 1:
        interviewer = speakers[0]
    roles = {speaker: ("interviewer" if speaker == interviewer else "candidate") for speaker in speakers}
    return roles


def load_topics(config: dict[str, Any]) -> dict[str, list[str]]:
    path = project_path(config, config["knowledge"]["topics_file"])
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return {str(label): [str(word) for word in words] for label, words in (payload.get("topics") or {}).items()}


def classify_topic(text: str, topics: dict[str, list[str]]) -> tuple[str, list[str]]:
    lowered = text.lower()
    hits: list[tuple[str, list[str]]] = []
    for label, words in topics.items():
        matched = [word for word in words if word.lower() in lowered]
        if matched:
            hits.append((label, matched))
    if not hits:
        return "未分类", []
    label, matched = max(hits, key=lambda item: len(item[1]))
    return label, matched


def _keyword_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _merge_speaker_turns(
    segments: list[dict[str, Any]], max_gap_seconds: float
) -> list[dict[str, Any]]:
    """Join punctuation-level segments that belong to one uninterrupted turn."""
    turns: list[dict[str, Any]] = []
    for segment in segments:
        member = dict(segment)
        member["segment_ids"] = [segment["id"]]
        if not turns:
            turns.append(member)
            continue

        previous = turns[-1]
        gap = float(segment["start"]) - float(previous["end"])
        if segment["speaker"] == previous["speaker"] and gap <= max_gap_seconds:
            previous["end"] = max(float(previous["end"]), float(segment["end"]))
            previous["text"] = f"{previous['text']}{segment['text']}".strip()
            previous["segment_ids"].append(segment["id"])
        else:
            turns.append(member)
    return turns


def _segment_ids(segment: dict[str, Any]) -> list[str]:
    return list(segment.get("segment_ids") or [segment["id"]])


def _new_qa(segment: dict[str, Any]) -> dict[str, Any]:
    segment_ids = _segment_ids(segment)
    return {
        "main_question": {"segment_ids": segment_ids, "text": segment["text"]},
        "follow_ups": [],
        "candidate_answers": [],
        "_active_prompt": "main",
        "_all_question_text": segment["text"],
        "_segment_ids": segment_ids[:],
    }


def _add_answer(qa: dict[str, Any], segment: dict[str, Any]) -> None:
    prompt = qa["_active_prompt"]
    member_ids = _segment_ids(segment)
    qa["_segment_ids"].extend(member_ids)
    answers = qa["candidate_answers"]
    if answers and answers[-1]["for_question"] == prompt:
        answers[-1]["segment_ids"].extend(member_ids)
        answers[-1]["text"] = f"{answers[-1]['text']} {segment['text']}".strip()
    else:
        answers.append(
            {"for_question": prompt, "segment_ids": member_ids, "text": segment["text"]}
        )


def _finish_qa(
    qa: dict[str, Any],
    completed: list[dict[str, Any]],
    topics: dict[str, list[str]],
    previous_keywords: set[str],
) -> set[str]:
    question_text = qa.pop("_all_question_text")
    qa.pop("_active_prompt", None)
    label, keywords = classify_topic(question_text, topics)
    current_keywords = {word.lower() for word in keywords}
    if not completed:
        relation = "first_topic"
    elif current_keywords and previous_keywords and current_keywords & previous_keywords:
        relation = "same_topic"
    elif label != "未分类" and completed[-1]["topic"]["label"] == label:
        relation = "same_topic"
    else:
        relation = "new_or_unknown_topic"
    qa["id"] = f"qa_{len(completed) + 1:03d}"
    qa["segment_ids"] = qa.pop("_segment_ids")
    qa["topic"] = {"label": label, "keywords": keywords, "relation_to_previous": relation}
    completed.append(qa)
    return current_keywords


def segment_qa(config: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
    segments = transcript.get("segments") or []
    qa_config = config["qa"]
    roles = infer_roles(segments, int(qa_config.get("minimum_question_score", 1)))
    for segment in segments:
        segment["role"] = roles.get(segment["speaker"], "unknown")
    for speaker in transcript.get("speakers", []):
        speaker["role"] = roles.get(speaker["id"], "unknown")

    turns = _merge_speaker_turns(
        segments, float(qa_config.get("speaker_turn_gap_seconds", 0.8))
    )
    topics = load_topics(config)
    threshold = float(qa_config.get("topic_similarity_threshold", 0.25))
    completed: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_keywords: set[str] = set()

    for segment in turns:
        text = segment["text"].strip()
        role = segment["role"]
        if role == "candidate":
            if current is not None:
                _add_answer(current, segment)
            continue
        if role != "interviewer" or text in ACKNOWLEDGEMENTS:
            continue

        if current is None:
            current = _new_qa(segment)
            continue

        current_label, current_words = classify_topic(current["_all_question_text"], topics)
        new_label, new_words = classify_topic(text, topics)
        same_label = current_label != "未分类" and current_label == new_label
        similarity = _keyword_similarity(
            {word.lower() for word in current_words}, {word.lower() for word in new_words}
        )
        is_follow_up = (
            not any(cue in text for cue in NEW_TOPIC_CUES)
            and (any(text.startswith(cue) for cue in FOLLOW_UP_CUES) or same_label or similarity >= threshold)
        )
        if is_follow_up:
            follow_up_id = f"follow_up_{len(current['follow_ups']) + 1}"
            current["follow_ups"].append(
                {"id": follow_up_id, "segment_ids": _segment_ids(segment), "text": text}
            )
            current["_segment_ids"].extend(_segment_ids(segment))
            current["_active_prompt"] = follow_up_id
            current["_all_question_text"] += " " + text
        else:
            previous_keywords = _finish_qa(current, completed, topics, previous_keywords)
            current = _new_qa(segment)

    if current is not None:
        _finish_qa(current, completed, topics, previous_keywords)
    transcript["qa_pairs"] = completed
    return transcript
