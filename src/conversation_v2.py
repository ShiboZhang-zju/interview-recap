from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import display_path
from .qa import ACKNOWLEDGEMENTS, classify_topic, load_topics
from .transcript import format_timestamp


SCHEMA_VERSION = "0.2.0"
VALID_ROLES = {"interviewer", "candidate", "unknown"}
VALID_PHASES = {
    "INTRO",
    "PROJECT",
    "TECH_QA",
    "CODING",
    "CODING_DISCUSSION",
    "BEHAVIORAL",
    "CANDIDATE_QUESTIONS",
    "END",
    "UNKNOWN",
}

MEANINGFUL_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
REPEATED_CHAR_RE = re.compile(r"([^\s，。！？、,.!?])\1{4,}")
QUESTION_WORD_RE = re.compile(
    r"为什么|怎么(?:做|处理|实现|看)?|如何|什么|哪(?:些|个|里)|多少|是否|有没有|能不能|"
    r"可不可以|介绍(?:一下)?|讲(?:一下|一讲|讲)?|说(?:一下|说)|请你|复杂度|区别|原理|"
    r"你会|你觉得|你认为|对吗|是吗|呢[？?]?"
)
INTERVIEWER_CONTROL_RE = re.compile(
    r"下一个问题|换一个问题|再问一个|接下来(?:问|聊|看)|我想问|(?:我)?有(?:几|个).*问题|"
    r"请你|你先|展开讲|具体讲|能讲|能说|说说|介绍一下"
)
CANDIDATE_EVIDENCE_RE = re.compile(
    r"我(?:负责|参与|做了|做的|设计|实现|优化|搭建|当时|会先|的思路|认为)|"
    r"我们(?:当时|项目|团队|会|把)|我的(?:工作|项目|思路|方案)|"
    r"首先我|然后我|当时我|具体来说|原因是|主要是"
)
FOLLOW_UP_RE = re.compile(
    r"^(那|那么|所以|具体|为什么|怎么|如果|刚才|进一步|还有|继续|这个|然后呢|复杂度)"
    r"|你刚才|刚刚提到|展开(?:讲|说)|细说|什么意思"
)
CLARIFICATION_RE = re.compile(
    r"什么意思|你是说|也就是说|换句话说|没听清|再说一遍|确认一下|我理解的是|"
    r"能看到吗|能听到吗"
)
NEW_TOPIC_RE = re.compile(
    r"下一个问题|换一个问题|再问一个|接下来|另外一个方向|最后一个问题|"
    r"再聊(?:一下)?|下面问"
)
ACK_RE = re.compile(r"^(嗯+|啊+|哦+|好(?:的)?|可以|明白|行|对+|是的|OK|ok)[，。！？、\s]*$")
ADMIN_RE = re.compile(
    r"能听到|听得到|摄像头|麦克风|共享屏幕|屏幕共享|网络问题|刷新(?:一下)?|"
    r"编辑器加载|会议.*重启|看得到吗|能看到吗|能看见|看得见"
)
CANDIDATE_INVITATION_RE = re.compile(
    r"你还有什么.*(?:想问|问题)|你有没有什么.*问题|你这边有什么.*问题|"
    r"我这边.*没有什么.*(?:问题|你)|还有什么想了解"
)
NO_MORE_QUESTIONS_RE = re.compile(r"(?:我)?(?:这)?边(?:没|没有)什么(?:问题)?")
PROMPT_MARKER_RE = re.compile(
    r"那你|那如果|那这个|我想问|你刚才|刚才你|你们会|会不会|能详细|能否|能不能|"
    r"请你|介绍一下|讲一讲|讲一下|说一下|复杂度呢|为什么|是怎么|是什么样|怎么样去"
)
HIGH_PRECISION_PROMPT_RE = re.compile(
    r"那你|那如果|我想问|你刚才|刚才你|你们会|会不会|能详细|能否|能不能|"
    r"请你|介绍一下|讲一讲|讲一下|说一下|复杂度呢"
)
DIRECT_ADDRESS_PROMPT_RE = re.compile(
    r"那你|那如果|我想问|你刚才|刚才你|刚刚.*(?:提到|说)|你们会|会不会|能详细|"
    r"请你|介绍一下|讲一讲|讲一下|说一下|复杂度呢|^为什么|你.*(?:为什么|怎么|如何)|"
    r"(?:区别|优劣|流程|原理|转移方程).*是什么|(?:区)?别是什么|优劣是.?什么|"
    r"转移方程.*为什么|那在.*是不是|是不是也有|是什么呢|项目.*部署|"
    r"那最后.*项目|那.*(?:decoder only|内存|显存)|"
    r"出现.*(?:case|错误|偏离).*(?:需要|处理|解决)|需要.*(?:处理|解决).*还[，。！？?]*$"
)
RESPONSE_START_RE = re.compile(
    r"^(好的|好|是的|对|嗯|你好|我叫|我(?:当时|负责|参与|做|会|先|的)|我们|首先|因为|可以用|我的)"
)
CODING_STRONG_RE = re.compile(
    r"开始(?:做|写).*题|看一下题目|先看题目|编辑器|编译器|在线编程|"
    r"代码的话.*(?:Python|Java|C\+?\+?|Go)|共享屏幕.*(?:题|代码)|"
    r"写一下代码|实现一下|给定一个|输入.*(?:数组|字符串)|输出.*(?:数组|结果)"
)
CODING_CONTEXT_RE = re.compile(
    r"题目|代码|数组|字符串|指针|下标|循环|递归|动态规划|DP|复杂度|编译|运行|"
    r"测试用例|边界|样例|输入|输出"
)
CODING_DISCUSSION_RE = re.compile(
    r"思路|复杂度|为什么|边界|优化|正确性|怎么想|解释一下|讲一下|这题|算法"
)
PROJECT_RE = re.compile(
    r"项目|实习|经历|负责|业务|公司|团队|工作里|工作中|做的.*(?:系统|平台|Agent|agent)"
)
PROJECT_PROMPT_RE = re.compile(
    r"展开.*(?:经历|项目)|讲讲.*(?:项目|经历)|介绍.*(?:项目|经历)|"
    r"你在.*(?:负责|做)|具体负责|挑战.*地方"
)
TECH_RE = re.compile(
    r"Redis|MySQL|HTTP|TCP|DPO|SFT|RAG|Transformer|Python|Java|GC|MVCC|"
    r"哈希|进程|线程|数据库|索引|网络|模型|算法"
)
BEHAVIORAL_RE = re.compile(
    r"最大的困难|遇到.*冲突|团队合作|如何沟通|为什么离职|优点|缺点|职业规划|"
    r"失败的经历"
)
CANDIDATE_QUESTION_RE = re.compile(
    r"你有没有什么.*问题|你这边有什么.*问题|关于我们.*岗位|关于这个岗位|"
    r"岗位.*具体.*(?:做|负责)|团队.*(?:方向|业务)|我想问一下.*(?:岗位|团队|业务)|"
    r"这边组里.*做什么"
)
CANDIDATE_QUESTION_START_RE = re.compile(
    r"我想问(?:一下)?.*(?:岗位|团队|业务|工作|方向|场景)|关于.*岗位.*(?:想问|了解)|岗位.*具体.*(?:做|负责)|"
    r"团队.*具体.*(?:方向|业务)"
)
CANDIDATE_QUESTION_TOPIC_RE = re.compile(r"岗位|团队|业务|工作内容|这个组|方向|场景")
END_RE = re.compile(r"今天面试先到|面试.*(?:结束|到这里)|拜拜|谢谢你.*拜|那今天就这样")


def _meaningful_chars(text: str) -> list[str]:
    return MEANINGFUL_RE.findall(text)


def _question_strength(text: str) -> float:
    compact = re.sub(r"\s+", "", text)
    meaningful_count = len(_meaningful_chars(compact))
    score = 0.0
    if re.search(r"[？?]", text):
        score += 2.0
    if INTERVIEWER_CONTROL_RE.search(text):
        score += 2.0
    starts_like_question = re.match(
        r"^(那|那么|所以|请|你|能|可以|为什么|怎么|如何|什么|哪(?!怕)|介绍|讲一下|讲讲|"
        r"说一下|说说|复杂度)",
        compact,
    )
    if starts_like_question and QUESTION_WORD_RE.search(compact):
        score += 1.5
    if re.search(r"(吗|么|呢|为什么|怎么|如何|什么)[，。！？?]*$", compact):
        score += 1.5
    if QUESTION_WORD_RE.search(compact):
        score += 1.0 if meaningful_count <= 30 else 0.2
    return score


def _is_acknowledgement(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return compact in ACKNOWLEDGEMENTS or bool(ACK_RE.fullmatch(compact))


def analyze_segment(segment: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Classify a segment without deleting or rewriting its source text."""
    text = str(segment.get("text") or "").strip()
    duration = max(0.0, float(segment.get("end", 0)) - float(segment.get("start", 0)))
    meaningful = _meaningful_chars(text)
    meaningful_count = len(meaningful)
    density = meaningful_count / max(duration, 0.001)
    question_strength = _question_strength(text)
    v2 = config.get("conversation_v2", {})
    long_seconds = float(v2.get("long_segment_seconds", 12.0))
    extreme_seconds = float(v2.get("extreme_segment_seconds", 25.0))
    density_threshold = float(v2.get("minimum_long_text_density", 0.5))
    reasons: list[str] = []

    if not meaningful:
        reasons.append("pure_or_repeated_punctuation")
    else:
        counts = Counter(char.lower() for char in meaningful)
        dominant_ratio = max(counts.values()) / meaningful_count
        if meaningful_count >= 8 and (REPEATED_CHAR_RE.search(text) or dominant_ratio >= 0.68):
            reasons.append("repetitive_noise_transcription")

        short_real_question = duration <= 8.0 and question_strength >= 2.0
        if not short_real_question:
            if duration >= extreme_seconds and density < density_threshold:
                reasons.append("long_duration_low_text_density")
            elif duration >= long_seconds and meaningful_count <= 5:
                reasons.append("long_duration_very_little_text")

    valid = not reasons
    return {
        "valid": valid,
        "reasons": reasons,
        "duration_seconds": round(duration, 3),
        "meaningful_char_count": meaningful_count,
        "text_density": round(density, 3),
        "question_strength": round(question_strength, 2),
    }


def _find_first_index(
    segments: list[dict[str, Any]], pattern: re.Pattern[str], minimum_time: float = 0.0
) -> int | None:
    for index, segment in enumerate(segments):
        if not segment["cleanup"]["valid"] or float(segment["start"]) < minimum_time:
            continue
        if pattern.search(segment["text"]):
            return index
    return None


def detect_phases(segments: list[dict[str, Any]], audio_duration: float) -> list[dict[str, Any]]:
    if not segments:
        return []

    project_start = _find_first_index(segments, PROJECT_PROMPT_RE, 45.0)
    contextual_project_start: int | None = None
    for index, segment in enumerate(segments):
        if not segment["cleanup"]["valid"] or float(segment["start"]) < 120.0:
            continue
        window_start = index
        while (
            window_start > 0
            and float(segment["end"]) - float(segments[window_start - 1]["start"]) <= 45.0
        ):
            window_start -= 1
        window_text = "".join(item["text"] for item in segments[window_start : index + 1])
        addressed_candidate = bool(
            re.search(r"你.*(?:项目|实习|经历|负责|做|迭代)|(?:项目|实习|经历).*你", window_text)
        )
        if (
            PROJECT_RE.search(window_text)
            and addressed_candidate
            and _question_strength(window_text) >= 2.0
        ):
            contextual_project_start = window_start
            break
    candidates = [index for index in (project_start, contextual_project_start) if index is not None]
    project_start = min(candidates) if candidates else None
    coding_start = _find_first_index(segments, CODING_STRONG_RE, 120.0)
    candidate_questions_start = _find_first_index(
        segments, CANDIDATE_QUESTION_START_RE, audio_duration * 0.5
    )
    if candidate_questions_start is None:
        for index, segment in enumerate(segments):
            if not segment["cleanup"]["valid"] or float(segment["start"]) < audio_duration * 0.5:
                continue
            window_start = index
            while (
                window_start > 0
                and float(segment["end"]) - float(segments[window_start - 1]["start"]) <= 20.0
            ):
                window_start -= 1
            window_text = "".join(item["text"] for item in segments[window_start : index + 1])
            if re.search(r"我想问(?:一下)?", window_text) and CANDIDATE_QUESTION_TOPIC_RE.search(
                window_text
            ):
                candidate_questions_start = window_start
                break
    end_start = _find_first_index(segments, END_RE, audio_duration * 0.65)

    def backtrack(index: int | None, seconds: float) -> int | None:
        if index is None:
            return None
        threshold = float(segments[index]["start"]) - seconds
        while index > 0 and float(segments[index - 1]["start"]) >= threshold:
            index -= 1
        return index

    project_start = backtrack(project_start, 10.0)
    coding_start = backtrack(coding_start, 10.0)

    active = "INTRO"
    for index, segment in enumerate(segments):
        text = segment["text"]
        if end_start is not None and index >= end_start:
            phase = "END"
        elif candidate_questions_start is not None and index >= candidate_questions_start:
            phase = "CANDIDATE_QUESTIONS"
        elif coding_start is not None and index >= coding_start:
            if CODING_DISCUSSION_RE.search(text) and _question_strength(text) >= 2.0:
                phase = "CODING_DISCUSSION"
            else:
                phase = "CODING"
        else:
            if project_start is not None and index >= project_start and active == "INTRO":
                active = "PROJECT"
            if BEHAVIORAL_RE.search(text):
                active = "BEHAVIORAL"
            elif PROJECT_PROMPT_RE.search(text):
                active = "PROJECT"
            elif TECH_RE.search(text) and _question_strength(text) >= 2.5 and active != "INTRO":
                active = "TECH_QA"
            phase = active
        segment["phase"] = phase if phase in VALID_PHASES else "UNKNOWN"

    spans: list[dict[str, Any]] = []
    for segment in segments:
        phase = segment["phase"]
        if spans and spans[-1]["phase"] == phase:
            spans[-1]["end"] = segment["end"]
            spans[-1]["segment_ids"].append(segment["id"])
        else:
            spans.append(
                {
                    "phase": phase,
                    "start": segment["start"],
                    "end": segment["end"],
                    "segment_ids": [segment["id"]],
                }
            )
    return spans


def _speaker_priors(segments: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    evidence: dict[str, dict[str, float]] = defaultdict(
        lambda: {"interviewer": 1.0, "candidate": 1.0}
    )
    for segment in segments:
        if not segment["cleanup"]["valid"]:
            continue
        speaker_id = segment["speaker_id"]
        text = segment["text"]
        chars = len(_meaningful_chars(text))
        prompt_strength = _question_strength(text)
        if prompt_strength >= 2.5:
            evidence[speaker_id]["interviewer"] += min(prompt_strength, 5.0) * 1.5
        if INTERVIEWER_CONTROL_RE.search(text):
            evidence[speaker_id]["interviewer"] += 3.0
        evidence[speaker_id]["candidate"] += chars / 45.0
        if CANDIDATE_EVIDENCE_RE.search(text):
            evidence[speaker_id]["candidate"] += 3.0
        if chars >= 60:
            evidence[speaker_id]["candidate"] += 2.0

    priors: dict[str, dict[str, float]] = {}
    interviewer_total = sum(values["interviewer"] for values in evidence.values())
    candidate_total = sum(values["candidate"] for values in evidence.values())
    for speaker_id, values in evidence.items():
        interviewer_share = values["interviewer"] / max(interviewer_total, 0.001)
        candidate_share = values["candidate"] / max(candidate_total, 0.001)
        interviewer_signal = interviewer_share + max(0.0, 0.5 - candidate_share)
        candidate_signal = candidate_share + max(0.0, 0.5 - interviewer_share)
        total = interviewer_signal + candidate_signal
        priors[speaker_id] = {
            "interviewer": round(interviewer_signal / total, 4),
            "candidate": round(candidate_signal / total, 4),
        }
    return priors


def _lexical_role_scores(text: str) -> tuple[float, float]:
    prompt_strength = _question_strength(text)
    interviewer = prompt_strength if prompt_strength >= 2.5 else 0.0
    candidate = 0.0
    chars = len(_meaningful_chars(text))
    if CANDIDATE_EVIDENCE_RE.search(text):
        candidate += 3.0
    if chars >= 30:
        candidate += 1.0
    if chars >= 80:
        candidate += 1.0
    if re.match(r"^(好的|好|是的|对|嗯|首先|我觉得|我的思路|当时|因为)", text):
        candidate += 1.0
    if INTERVIEWER_CONTROL_RE.search(text) and prompt_strength >= 2.5:
        interviewer += 1.5
    return interviewer, candidate


def infer_segment_roles(
    segments: list[dict[str, Any]], priors: dict[str, dict[str, float]]
) -> None:
    expected_answer_role: str | None = None
    previous_role = "unknown"
    previous_speaker = ""
    previous_end = 0.0
    last_prompt_speaker = ""

    for segment in segments:
        if not segment["cleanup"]["valid"]:
            segment["role"] = "unknown"
            segment["role_confidence"] = 0.0
            continue

        text = segment["text"]
        speaker_id = segment["speaker_id"]
        interviewer, candidate = _lexical_role_scores(text)
        prior = priors.get(speaker_id, {"interviewer": 0.5, "candidate": 0.5})
        interviewer += prior["interviewer"] * 2.0
        candidate += prior["candidate"] * 2.0
        question_like = _question_strength(text) >= 2.5

        if (
            expected_answer_role == "candidate"
            and question_like
            and not re.search(r"[？?]", text)
            and not DIRECT_ADDRESS_PROMPT_RE.search(text)
        ):
            candidate += 2.0

        if segment["phase"] == "CANDIDATE_QUESTIONS" and question_like:
            candidate += 4.0
            interviewer -= 1.0
        elif (
            expected_answer_role
            and not question_like
            and (
                speaker_id != last_prompt_speaker
                or CANDIDATE_EVIDENCE_RE.search(text)
                or len(_meaningful_chars(text)) >= 35
            )
        ):
            if expected_answer_role == "candidate":
                candidate += 1.25
            else:
                interviewer += 1.25

        gap = float(segment["start"]) - previous_end
        if speaker_id == previous_speaker and gap <= 1.0 and previous_role in VALID_ROLES:
            if previous_role == "interviewer":
                interviewer += 0.4
            elif previous_role == "candidate":
                candidate += 0.4

        difference = abs(interviewer - candidate)
        maximum = max(interviewer, candidate)
        prior_difference = abs(prior["interviewer"] - prior["candidate"])
        if maximum < 0.75 or (difference < 0.2 and prior_difference < 0.08):
            role = "unknown"
            confidence = 0.45
        elif interviewer > candidate:
            role = "interviewer"
            confidence = min(0.98, 0.54 + difference * 0.09 + prior_difference * 0.15)
        else:
            role = "candidate"
            confidence = min(0.98, 0.54 + difference * 0.09 + prior_difference * 0.15)

        segment["role"] = role
        segment["role_confidence"] = round(confidence, 2)
        if question_like and role == "interviewer":
            expected_answer_role = "candidate"
            last_prompt_speaker = speaker_id
        elif question_like and role == "candidate" and segment["phase"] == "CANDIDATE_QUESTIONS":
            expected_answer_role = "interviewer"
            last_prompt_speaker = speaker_id
        previous_role = role
        previous_speaker = speaker_id
        previous_end = float(segment["end"])


def _phase_family(phase: str) -> str:
    if phase in {"CODING", "CODING_DISCUSSION"}:
        return "CODING"
    if phase in {"PROJECT", "TECH_QA"}:
        return "INTERVIEW_QA"
    return phase


def _build_turns(segments: list[dict[str, Any]], max_gap: float) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for segment in segments:
        if not segment["cleanup"]["valid"]:
            continue
        member = {
            "start": segment["start"],
            "end": segment["end"],
            "speaker_ids": [segment["speaker_id"]],
            "role": segment["role"],
            "role_confidence": segment["role_confidence"],
            "phase": segment["phase"],
            "text": segment["text"],
            "segment_ids": [segment["id"]],
        }
        if not turns:
            turns.append(member)
            continue

        previous = turns[-1]
        gap = float(segment["start"]) - float(previous["end"])
        compatible_phase = _phase_family(previous["phase"]) == _phase_family(segment["phase"])
        same_role = previous["role"] == segment["role"] and segment["role"] != "unknown"
        attach_unknown = (
            segment["role"] == "unknown"
            and previous["role"] != "unknown"
            and gap <= min(0.7, max_gap)
            and _question_strength(segment["text"]) < 2.0
        )
        if compatible_phase and gap <= max_gap and (same_role or attach_unknown):
            previous["end"] = max(float(previous["end"]), float(segment["end"]))
            previous["text"] = f"{previous['text']}{segment['text']}".strip()
            previous["segment_ids"].append(segment["id"])
            if segment["speaker_id"] not in previous["speaker_ids"]:
                previous["speaker_ids"].append(segment["speaker_id"])
            previous["role_confidence"] = round(
                min(previous["role_confidence"], segment["role_confidence"]), 2
            )
        else:
            turns.append(member)
    return turns


def _is_question_turn(turn: dict[str, Any]) -> bool:
    text = turn["text"].strip()
    if (
        _is_acknowledgement(text)
        or ADMIN_RE.search(text)
        or CANDIDATE_INVITATION_RE.search(text)
        or (
            NO_MORE_QUESTIONS_RE.search(re.sub(r"\s+", "", text))
            and not DIRECT_ADDRESS_PROMPT_RE.search(text)
        )
    ):
        return False
    if turn["phase"] == "END":
        return False
    strength = _question_strength(text)
    if turn["phase"] == "CANDIDATE_QUESTIONS":
        return turn["role"] == "candidate" and strength >= 2.5
    if turn["role"] == "interviewer":
        explicit_question = bool(re.search(r"[？?]", text)) and len(_meaningful_chars(text)) >= 4
        return strength >= 2.5 or (explicit_question and strength >= 2.0)
    return turn["role"] == "unknown" and strength >= 4.0


def _question_relation(
    turn: dict[str, Any],
    current_root: dict[str, Any] | None,
    topics: dict[str, list[str]],
) -> tuple[str, str]:
    if current_root is None:
        return "main_question", "first_question"
    text = turn["text"]
    if turn["role"] != current_root["asked_by"]:
        return "main_question", "topic_switch"
    if _phase_family(turn["phase"]) != _phase_family(current_root["phase"]):
        return "main_question", "topic_switch"
    if NEW_TOPIC_RE.search(text):
        return "main_question", "topic_switch"
    if CLARIFICATION_RE.search(text):
        return "clarification", "same_topic"
    if FOLLOW_UP_RE.search(text) or len(_meaningful_chars(text)) <= 16:
        return "follow_up", "same_topic"

    root_label, _ = classify_topic(current_root["text"], topics)
    new_label, _ = classify_topic(text, topics)
    if root_label != "未分类" and new_label != "未分类" and root_label != new_label:
        return "main_question", "topic_switch"
    if float(turn["start"]) - float(current_root["start"]) >= 240.0:
        return "main_question", "topic_switch"
    if re.match(r"^(请|介绍|说说|讲讲)", text) and current_root.get("children"):
        return "main_question", "topic_switch"
    return "follow_up", "same_topic"


def _new_question_node(
    node_id: str,
    node_type: str,
    relation: str,
    turn: dict[str, Any],
    topics: dict[str, list[str]],
) -> dict[str, Any]:
    label, keywords = classify_topic(turn["text"], topics)
    return {
        "id": node_id,
        "node_type": node_type,
        "relation_to_previous": relation,
        "asked_by": turn["role"],
        "role_confidence": turn["role_confidence"],
        "phase": turn["phase"],
        "start": turn["start"],
        "end": turn["end"],
        "speaker_ids": turn["speaker_ids"],
        "segment_ids": turn["segment_ids"],
        "text": turn["text"],
        "topic": {"label": label, "keywords": keywords},
        "answers": [],
        "children": [],
    }


def _add_answer(node: dict[str, Any], turn: dict[str, Any]) -> None:
    answers = node["answers"]
    if answers and answers[-1]["role"] == turn["role"]:
        answers[-1]["end"] = turn["end"]
        answers[-1]["text"] = f"{answers[-1]['text']} {turn['text']}".strip()
        answers[-1]["segment_ids"].extend(turn["segment_ids"])
        for speaker_id in turn["speaker_ids"]:
            if speaker_id not in answers[-1]["speaker_ids"]:
                answers[-1]["speaker_ids"].append(speaker_id)
    else:
        answers.append(
            {
                "role": turn["role"],
                "role_confidence": turn["role_confidence"],
                "start": turn["start"],
                "end": turn["end"],
                "speaker_ids": turn["speaker_ids"],
                "segment_ids": turn["segment_ids"],
                "text": turn["text"],
            }
        )


def _prompt_anchor(segment: dict[str, Any]) -> tuple[bool, str]:
    if not segment["cleanup"]["valid"] or segment["phase"] == "END":
        return False, "unknown"
    text = segment["text"].strip()
    compact = re.sub(r"\s+", "", text)
    if (
        _is_acknowledgement(text)
        or ADMIN_RE.search(text)
        or CANDIDATE_INVITATION_RE.search(text)
        or (NO_MORE_QUESTIONS_RE.search(compact) and not DIRECT_ADDRESS_PROMPT_RE.search(text))
    ):
        return False, "unknown"

    strength = _question_strength(text)
    marker = bool(PROMPT_MARKER_RE.search(text))
    precise_marker = bool(HIGH_PRECISION_PROMPT_RE.search(text))
    direct_marker = bool(DIRECT_ADDRESS_PROMPT_RE.search(text))
    explicit = (
        bool(re.search(r"[？?]", text))
        and len(_meaningful_chars(text)) >= 4
        and (bool(QUESTION_WORD_RE.search(text)) or direct_marker)
    )

    if segment["phase"] == "CANDIDATE_QUESTIONS":
        is_anchor = (strength >= 2.5 or marker) and not CANDIDATE_INVITATION_RE.search(text)
        return is_anchor, "candidate" if is_anchor else "unknown"

    if _phase_family(segment["phase"]) == "CODING":
        is_anchor = strength >= 3.0 or direct_marker
    elif segment["role"] == "interviewer":
        is_anchor = strength >= 2.5 or (explicit and strength >= 2.0) or direct_marker
    else:
        is_anchor = direct_marker or (precise_marker and strength >= 2.0)
    return is_anchor, "interviewer" if is_anchor else "unknown"


def _build_prompt_spans(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: list[tuple[int, str]] = []
    for index, segment in enumerate(segments):
        is_anchor, asked_by = _prompt_anchor(segment)
        if (
            is_anchor
            and index > 0
            and re.search(r"比如(?:说)?|例如|举个例子", segments[index - 1]["text"])
            and not DIRECT_ADDRESS_PROMPT_RE.search(segment["text"])
        ):
            is_anchor = False
        if is_anchor:
            anchors.append((index, asked_by))

    spans: list[dict[str, Any]] = []
    anchor_indexes = {index for index, _ in anchors}
    for index, asked_by in anchors:
        segment = segments[index]
        start_index = index
        if index > 0:
            previous = segments[index - 1]
            gap = float(segment["start"]) - float(previous["end"])
            if (
                previous["cleanup"]["valid"]
                and index - 1 not in anchor_indexes
                and _phase_family(previous["phase"]) == _phase_family(segment["phase"])
                and gap <= 0.8
                and previous["role"] == asked_by
                and not RESPONSE_START_RE.search(previous["text"])
            ):
                start_index = index - 1

        end_index = index
        has_terminal_question = bool(re.search(r"[？?]", segment["text"]))
        while not has_terminal_question and end_index + 1 < len(segments):
            next_index = end_index + 1
            next_segment = segments[next_index]
            gap = float(next_segment["start"]) - float(segments[end_index]["end"])
            elapsed = float(next_segment["end"]) - float(segment["start"])
            if (
                next_index in anchor_indexes
                or not next_segment["cleanup"]["valid"]
                or _phase_family(next_segment["phase"]) != _phase_family(segment["phase"])
                or gap > 1.2
                or elapsed > 8.0
                or RESPONSE_START_RE.search(next_segment["text"])
                or (
                    next_segment["role"] != asked_by
                    and float(next_segment["role_confidence"]) >= 0.6
                    and next_segment["speaker_id"] != segment["speaker_id"]
                )
            ):
                break
            end_index = next_index
            has_terminal_question = bool(re.search(r"[？?]", next_segment["text"]))

        members = segments[start_index : end_index + 1]
        span = {
            "start": members[0]["start"],
            "end": members[-1]["end"],
            "speaker_ids": list(dict.fromkeys(item["speaker_id"] for item in members)),
            "role": asked_by,
            "role_confidence": round(
                max(0.55, max(float(item["role_confidence"]) for item in members)), 2
            ),
            "phase": segment["phase"],
            "text": "".join(item["text"] for item in members),
            "segment_ids": [item["id"] for item in members],
        }
        if spans and float(span["start"]) <= float(spans[-1]["end"]) + 0.5:
            previous = spans[-1]
            if previous["role"] == span["role"] and _phase_family(previous["phase"]) == _phase_family(
                span["phase"]
            ):
                previous["end"] = max(float(previous["end"]), float(span["end"]))
                previous["segment_ids"] = list(
                    dict.fromkeys(previous["segment_ids"] + span["segment_ids"])
                )
                previous["speaker_ids"] = list(
                    dict.fromkeys(previous["speaker_ids"] + span["speaker_ids"])
                )
                member_ids = set(previous["segment_ids"])
                previous["text"] = "".join(
                    item["text"] for item in segments if item["id"] in member_ids
                )
                continue
        spans.append(span)
    return spans


def _answer_between(
    segments: list[dict[str, Any]],
    prompt: dict[str, Any],
    next_prompt: dict[str, Any] | None,
) -> dict[str, Any] | None:
    upper_bound = float(next_prompt["start"]) if next_prompt else float("inf")
    prompt_ids = set(prompt["segment_ids"])
    members = [
        segment
        for segment in segments
        if segment["cleanup"]["valid"]
        and segment["id"] not in prompt_ids
        and float(segment["start"]) >= float(prompt["end"])
        and float(segment["start"]) < upper_bound
        and segment["phase"] != "END"
        and not _is_acknowledgement(segment["text"])
        and not ADMIN_RE.search(segment["text"])
    ]
    if not members:
        return None
    expected_role = "candidate" if prompt["role"] == "interviewer" else "interviewer"
    return {
        "role": expected_role,
        "role_confidence": round(
            sum(float(item["role_confidence"]) for item in members) / len(members), 2
        ),
        "start": members[0]["start"],
        "end": members[-1]["end"],
        "speaker_ids": list(dict.fromkeys(item["speaker_id"] for item in members)),
        "segment_ids": [item["id"] for item in members],
        "text": "".join(item["text"] for item in members),
    }


def build_question_tree(
    config: dict[str, Any], segments: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_gap = float(config.get("conversation_v2", {}).get("turn_gap_seconds", 1.2))
    turns = _build_turns(segments, max_gap)
    prompts = _build_prompt_spans(segments)
    topics = load_topics(config)
    roots: list[dict[str, Any]] = []
    current_root: dict[str, Any] | None = None

    for index, prompt in enumerate(prompts):
        node_type, relation = _question_relation(prompt, current_root, topics)
        if node_type == "main_question" or current_root is None:
            root_id = f"Q{len(roots) + 1}"
            node = _new_question_node(root_id, "main_question", relation, prompt, topics)
            roots.append(node)
            current_root = node
        else:
            child_id = f"{current_root['id']}.{len(current_root['children']) + 1}"
            node = _new_question_node(child_id, node_type, relation, prompt, topics)
            current_root["children"].append(node)

        next_prompt = prompts[index + 1] if index + 1 < len(prompts) else None
        answer = _answer_between(segments, prompt, next_prompt)
        if answer is not None:
            node["answers"].append(answer)

    return roots, turns


def _merge_events(events: list[dict[str, Any]], maximum_gap: float = 3.0) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (item["start"], item["end"])):
        if (
            merged
            and merged[-1]["event_type"] == event["event_type"]
            and float(event["start"]) <= float(merged[-1]["end"]) + maximum_gap
        ):
            merged[-1]["end"] = max(float(merged[-1]["end"]), float(event["end"]))
            merged[-1]["duration_seconds"] = round(
                float(merged[-1]["end"]) - float(merged[-1]["start"]), 3
            )
            merged[-1]["segment_ids"].extend(event.get("segment_ids", []))
            merged[-1]["reasons"] = sorted(
                set(merged[-1].get("reasons", [])) | set(event.get("reasons", []))
            )
        else:
            merged.append(copy.deepcopy(event))
    for index, event in enumerate(merged, start=1):
        event["id"] = f"event_{index:03d}"
    return merged


def build_events(config: dict[str, Any], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    v2 = config.get("conversation_v2", {})
    silence_threshold = float(v2.get("coding_silence_min_seconds", 8.0))
    events: list[dict[str, Any]] = []

    for segment in segments:
        if _phase_family(segment["phase"]) != "CODING" or segment["cleanup"]["valid"]:
            continue
        reasons = segment["cleanup"]["reasons"]
        event_type = (
            "coding_low_speech_density"
            if any(reason.startswith("long_duration") for reason in reasons)
            else "coding_noise"
        )
        events.append(
            {
                "event_type": event_type,
                "phase": segment["phase"],
                "start": segment["start"],
                "end": segment["end"],
                "duration_seconds": round(float(segment["end"]) - float(segment["start"]), 3),
                "segment_ids": [segment["id"]],
                "reasons": reasons,
            }
        )

    for left, right in zip(segments, segments[1:]):
        if _phase_family(left["phase"]) != "CODING" or _phase_family(right["phase"]) != "CODING":
            continue
        gap = float(right["start"]) - float(left["end"])
        if gap >= silence_threshold:
            events.append(
                {
                    "event_type": "coding_silence",
                    "phase": "CODING",
                    "start": left["end"],
                    "end": right["start"],
                    "duration_seconds": round(gap, 3),
                    "segment_ids": [left["id"], right["id"]],
                    "reasons": ["gap_between_coding_segments"],
                }
            )
    return _merge_events(events)


def _flatten_question_tree(roots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for root in roots:
        nodes = [root, *root["children"]]
        segment_ids: list[str] = []
        candidate_answers: list[dict[str, Any]] = []
        for node in nodes:
            segment_ids.extend(node["segment_ids"])
            for answer in node["answers"]:
                segment_ids.extend(answer["segment_ids"])
                if answer["role"] == "candidate":
                    candidate_answers.append(
                        {
                            "for_question": node["id"],
                            "segment_ids": answer["segment_ids"],
                            "text": answer["text"],
                        }
                    )
        flattened.append(
            {
                "id": f"qa_v2_{len(flattened) + 1:03d}",
                "root_question_id": root["id"],
                "main_question": {
                    "segment_ids": root["segment_ids"],
                    "text": root["text"],
                },
                "follow_ups": [
                    {
                        "id": child["id"],
                        "type": child["node_type"],
                        "segment_ids": child["segment_ids"],
                        "text": child["text"],
                    }
                    for child in root["children"]
                ],
                "candidate_answers": candidate_answers,
                "segment_ids": list(dict.fromkeys(segment_ids)),
                "topic": root["topic"],
                "relation_to_previous": root["relation_to_previous"],
            }
        )
    return flattened


def _statistics(
    original: dict[str, Any],
    segments: list[dict[str, Any]],
    roots: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes = [node for root in roots for node in [root, *root["children"]]]
    invalid = [segment for segment in segments if not segment["cleanup"]["valid"]]
    valid = [segment for segment in segments if segment["cleanup"]["valid"]]
    cleanup_reasons = Counter(
        reason for segment in invalid for reason in segment["cleanup"]["reasons"]
    )
    old_roles = {segment["id"]: segment.get("role", "unknown") for segment in original.get("segments", [])}
    role_corrections = sum(
        old_roles.get(segment["id"], "unknown") in {"interviewer", "candidate"}
        and segment["role"] in {"interviewer", "candidate"}
        and old_roles[segment["id"]] != segment["role"]
        for segment in valid
    )
    return {
        "raw_segment_count": len(segments),
        "valid_segment_count": len(valid),
        "invalid_segment_count": len(invalid),
        "v1_qa_count": len(original.get("qa_pairs") or []),
        "qa_count": len(roots),
        "question_count": len(nodes),
        "main_question_count": len(roots),
        "follow_up_count": sum(node["node_type"] == "follow_up" for node in nodes),
        "clarification_count": sum(node["node_type"] == "clarification" for node in nodes),
        "topic_switch_count": sum(
            root["relation_to_previous"] == "topic_switch" for root in roots
        ),
        "coding_silence_event_count": sum(
            event["event_type"] == "coding_silence" for event in events
        ),
        "coding_event_count": len(events),
        "unknown_role_count": sum(segment["role"] == "unknown" for segment in segments),
        "valid_unknown_role_count": sum(segment["role"] == "unknown" for segment in valid),
        "role_correction_count": role_corrections,
        "cleanup_reason_counts": dict(sorted(cleanup_reasons.items())),
    }


def repair_conversation(config: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
    """Create an auditable V2 derivative without mutating the V1 transcript."""
    original = copy.deepcopy(transcript)
    segments = copy.deepcopy(transcript.get("segments") or [])
    for segment in segments:
        segment["speaker_id"] = segment.get("speaker", "speaker_unknown")
        segment["cleanup"] = analyze_segment(segment, config)
        segment["v1_role"] = segment.get("role", "unknown")

    audio_duration = float((transcript.get("audio") or {}).get("duration") or 0.0)
    phase_spans = detect_phases(segments, audio_duration)
    priors = _speaker_priors(segments)
    infer_segment_roles(segments, priors)
    roots, turns = build_question_tree(config, segments)
    events = build_events(config, segments)
    qa_pairs = _flatten_question_tree(roots)

    speaker_summaries = []
    for speaker_id in sorted({segment["speaker_id"] for segment in segments}):
        members = [
            segment
            for segment in segments
            if segment["speaker_id"] == speaker_id and segment["cleanup"]["valid"]
        ]
        role_counts = Counter(segment["role"] for segment in members)
        prior = priors.get(speaker_id, {})
        dominant_role = max(prior, key=prior.get) if prior else "unknown"
        speaker_summaries.append(
            {
                "id": speaker_id,
                "dominant_role": dominant_role,
                "role_confidence": round(prior.get(dominant_role, 0.0), 2),
                "role_counts": dict(role_counts),
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "derived_from": {
            "schema_version": transcript.get("schema_version"),
            "source_audio": transcript.get("source_audio"),
            "raw_output_preserved": bool(
                (transcript.get("pipeline") or {}).get("raw_output_preserved", False)
            ),
        },
        "source_audio": transcript.get("source_audio"),
        "audio": copy.deepcopy(transcript.get("audio")),
        "pipeline": copy.deepcopy(transcript.get("pipeline")),
        "preprocessing": copy.deepcopy(transcript.get("preprocessing")),
        "conversation_structure": {
            "version": "V2",
            "method": "deterministic_local_heuristics",
            "speaker_diarization_is_weak_evidence": True,
            "valid_roles": sorted(VALID_ROLES),
            "valid_phases": sorted(VALID_PHASES),
        },
        "speakers": speaker_summaries,
        "speaker_role_priors": priors,
        "segments": segments,
        "phase_spans": phase_spans,
        "events": events,
        "turns": turns,
        "question_tree": roots,
        "qa_pairs": qa_pairs,
    }
    result["statistics"] = _statistics(original, segments, roots, events)
    return result


def v2_output_path(v1_path: str | Path) -> Path:
    path = Path(v1_path).expanduser().resolve()
    stem = path.stem[:-3] if path.stem.endswith(".v2") else path.stem
    return path.with_name(f"{stem}.v2.json")


def render_v2_markdown(transcript: dict[str, Any]) -> str:
    stats = transcript["statistics"]
    lines = ["# Conversation Structure V2", ""]
    lines.append(f"- 音频：`{transcript.get('source_audio') or 'unknown'}`")
    lines.append(
        f"- Segments：{stats['raw_segment_count']} raw / {stats['valid_segment_count']} valid"
    )
    lines.append(
        f"- Question Tree：{stats['main_question_count']} main / "
        f"{stats['follow_up_count']} follow-up / {stats['clarification_count']} clarification"
    )
    lines.append(f"- Coding events：{stats['coding_event_count']}")
    lines.extend(["", "## Phase spans", ""])
    for span in transcript.get("phase_spans", []):
        stamp = f"{format_timestamp(span['start'])}–{format_timestamp(span['end'])}"
        lines.append(f"- {stamp} · {span['phase']} · {len(span['segment_ids'])} segments")

    events = transcript.get("events") or []
    if events:
        lines.extend(["", "## Events", ""])
        for event in events:
            stamp = f"{format_timestamp(event['start'])}–{format_timestamp(event['end'])}"
            lines.append(
                f"- {event['id']} · {stamp} · {event['event_type']} · "
                f"{event['duration_seconds']:.1f}s · {', '.join(event.get('segment_ids', []))}"
            )

    lines.extend(["", "## Cleaned transcript", ""])
    for segment in transcript.get("segments", []):
        if not segment["cleanup"]["valid"]:
            continue
        stamp = f"{format_timestamp(segment['start'])}–{format_timestamp(segment['end'])}"
        lines.append(
            f"**[{stamp}] {segment['speaker_id']} · {segment['role']} "
            f"({segment['role_confidence']:.2f}) · {segment['phase']}**  "
        )
        lines.append(segment["text"])
        lines.append("")

    lines.extend(["## Question Tree", ""])
    for root in transcript.get("question_tree", []):
        lines.append(
            f"### {root['id']} · {root['node_type']} · {root['phase']} · "
            f"{root['topic']['label']}"
        )
        lines.append("")
        lines.append(f"- asked_by: {root['asked_by']} ({root['role_confidence']:.2f})")
        lines.append(f"- relation: {root['relation_to_previous']}")
        lines.append(f"- segment_ids: {', '.join(root['segment_ids'])}")
        lines.append(f"- question: {root['text']}")
        for answer in root["answers"]:
            lines.append(
                f"- answer [{answer['role']}] ({', '.join(answer['segment_ids'])}): "
                f"{answer['text']}"
            )
        for child in root["children"]:
            lines.append(
                f"  - {child['id']} · {child['node_type']} · "
                f"{child['text']} [{', '.join(child['segment_ids'])}]"
            )
            for answer in child["answers"]:
                lines.append(
                    f"    - answer [{answer['role']}]: {answer['text']} "
                    f"[{', '.join(answer['segment_ids'])}]"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_repaired(transcript: dict[str, Any], json_path: str | Path) -> tuple[Path, Path]:
    path = Path(json_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = path.with_suffix(".md")
    markdown_path.write_text(render_v2_markdown(transcript), encoding="utf-8")
    return path, markdown_path


def repair_file(
    config: dict[str, Any], structured_path: str | Path
) -> tuple[dict[str, Any], Path, Path]:
    source = Path(structured_path).expanduser().resolve()
    transcript = json.loads(source.read_text(encoding="utf-8"))
    repaired = repair_conversation(config, transcript)
    repaired["derived_from"]["structured_path"] = display_path(config, source)
    json_path, markdown_path = save_repaired(repaired, v2_output_path(source))
    return repaired, json_path, markdown_path
