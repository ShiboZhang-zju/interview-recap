from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .redaction import Redactor


ANALYSIS_SCHEMA_VERSION = "0.1.0"
SCORE_NAMES = ("relevance", "technical_depth", "clarity", "structure")


class AnalysisError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_loopback_url(value: str) -> bool:
    hostname = (urlparse(value).hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _require_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise AnalysisError(
            '未安装可选分析依赖；请运行 python -m pip install -e ".[analysis]"'
        ) from exc
    return httpx


def _parse_json_content(content: str) -> dict[str, Any]:
    value = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Provider 返回的内容不是有效 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AnalysisError("Provider 返回的 JSON 顶层必须是 object")
    return parsed


def _flatten_questions(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for node in nodes:
        flattened.append(node)
        flattened.extend(_flatten_questions(node.get("children") or []))
    return flattened


def build_minimized_payload(
    conversation: dict[str, Any],
    *,
    redactor: Redactor,
    redact: bool = True,
    maximum_answer_chars: int = 2400,
) -> dict[str, Any]:
    """Build the only payload providers may receive; audio and full segments are excluded."""

    if maximum_answer_chars < 1:
        raise AnalysisError("analysis.maximum_answer_chars 必须大于 0")

    def protected_text(value: Any, maximum_chars: int | None = None) -> str:
        text = str(value or "")
        if redact:
            text = redactor.redact_text(text)
        return text[:maximum_chars] if maximum_chars is not None else text

    questions: list[dict[str, Any]] = []
    for node in _flatten_questions(conversation.get("question_tree") or []):
        answers = []
        for answer in node.get("answers") or []:
            answers.append(
                {
                    "role": answer.get("role", "unknown"),
                    "text": protected_text(answer.get("text"), maximum_answer_chars),
                    "segment_ids": list(answer.get("segment_ids") or []),
                }
            )
        questions.append(
            {
                "question_id": node["id"],
                "node_type": node.get("node_type", "main_question"),
                "phase": node.get("phase", "UNKNOWN"),
                "topic": protected_text((node.get("topic") or {}).get("label", "未分类")),
                "question": protected_text(node.get("text")),
                "question_segment_ids": list(node.get("segment_ids") or []),
                "answers": answers,
            }
        )

    statistics = conversation.get("statistics") or {}
    payload: dict[str, Any] = {
        "conversation_schema_version": conversation.get("schema_version"),
        "statistics": {
            key: statistics.get(key, 0)
            for key in (
                "valid_segment_count",
                "main_question_count",
                "follow_up_count",
                "clarification_count",
                "coding_event_count",
                "unknown_role_count",
            )
        },
        "questions": questions,
    }
    return payload


def _question_evidence(conversation: dict[str, Any]) -> dict[str, set[str]]:
    evidence: dict[str, set[str]] = {}
    for node in _flatten_questions(conversation.get("question_tree") or []):
        ids = set(node.get("segment_ids") or [])
        for answer in node.get("answers") or []:
            ids.update(answer.get("segment_ids") or [])
        evidence[node["id"]] = ids
    return evidence


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AnalysisError(f"analysis 字段 {field} 必须是 string array")
    return value


def normalize_provider_result(
    result: dict[str, Any],
    conversation: dict[str, Any],
    *,
    evaluated: bool,
    expected_question_ids: set[str] | None = None,
) -> dict[str, Any]:
    overview = result.get("overview")
    if not isinstance(overview, dict):
        raise AnalysisError("Provider 结果缺少 overview object")
    normalized_overview = {
        "assessment_status": "evaluated" if evaluated else "not_evaluated",
        "summary": str(overview.get("summary") or ""),
        "strengths": _string_list(overview.get("strengths", []), "overview.strengths"),
        "risks": _string_list(overview.get("risks", []), "overview.risks"),
        "suggested_focus": _string_list(
            overview.get("suggested_focus", []), "overview.suggested_focus"
        ),
    }

    allowed_evidence = _question_evidence(conversation)
    expected = expected_question_ids or set(allowed_evidence)
    received_questions = result.get("questions")
    if not isinstance(received_questions, list):
        raise AnalysisError("Provider 结果缺少 questions array")
    normalized_questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(received_questions):
        if not isinstance(item, dict):
            raise AnalysisError(f"questions[{index}] 必须是 object")
        question_id = str(item.get("question_id") or "")
        if question_id not in allowed_evidence:
            raise AnalysisError(f"Provider 引用了未知 question_id: {question_id or 'empty'}")
        if question_id not in expected:
            raise AnalysisError(f"Provider 返回了当前批次之外的问题: {question_id}")
        if question_id in seen:
            raise AnalysisError(f"Provider 重复返回 question_id: {question_id}")
        seen.add(question_id)
        evidence_ids = _string_list(
            item.get("evidence_segment_ids", []),
            f"questions[{index}].evidence_segment_ids",
        )
        unknown_evidence = set(evidence_ids) - allowed_evidence[question_id]
        if unknown_evidence:
            raise AnalysisError(
                f"{question_id} 引用了不属于该问答的 segment: {', '.join(sorted(unknown_evidence))}"
            )
        raw_scores = item.get("scores") or {}
        if not isinstance(raw_scores, dict):
            raise AnalysisError(f"{question_id}.scores 必须是 object")
        scores: dict[str, int | None] = {}
        for name in SCORE_NAMES:
            score = raw_scores.get(name)
            if score is not None and (
                isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5
            ):
                raise AnalysisError(f"{question_id}.scores.{name} 必须为 1-5 或 null")
            scores[name] = score
        improved = item.get("improved_answer")
        if improved is not None and not isinstance(improved, str):
            raise AnalysisError(f"{question_id}.improved_answer 必须为 string 或 null")
        normalized_questions.append(
            {
                "question_id": question_id,
                "assessment_status": "evaluated" if evaluated else "not_evaluated",
                "summary": str(item.get("summary") or ""),
                "scores": scores,
                "strengths": _string_list(item.get("strengths", []), f"{question_id}.strengths"),
                "risks": _string_list(item.get("risks", []), f"{question_id}.risks"),
                "suggestions": _string_list(
                    item.get("suggestions", []), f"{question_id}.suggestions"
                ),
                "improved_answer": improved,
                "evidence_segment_ids": evidence_ids,
            }
        )

    missing = expected - seen
    if missing:
        raise AnalysisError(f"Provider 未返回问题: {', '.join(sorted(missing))}")

    raw_plan = result.get("review_plan", [])
    if not isinstance(raw_plan, list):
        raise AnalysisError("review_plan 必须是 array")
    review_plan: list[dict[str, Any]] = []
    for index, item in enumerate(raw_plan):
        if not isinstance(item, dict):
            raise AnalysisError(f"review_plan[{index}] 必须是 object")
        question_ids = _string_list(item.get("question_ids", []), "review_plan.question_ids")
        unknown_questions = set(question_ids) - expected
        if unknown_questions:
            raise AnalysisError(
                f"review_plan 引用了未知问题: {', '.join(sorted(unknown_questions))}"
            )
        review_plan.append(
            {
                "priority": str(item.get("priority") or "medium"),
                "topic": str(item.get("topic") or ""),
                "action": str(item.get("action") or ""),
                "question_ids": question_ids,
            }
        )

    return {
        "overview": normalized_overview,
        "questions": normalized_questions,
        "review_plan": review_plan,
        "warnings": _string_list(result.get("warnings", []), "warnings"),
    }


class AnalysisProvider(ABC):
    name = "unknown"
    remote = False

    def __init__(self, model: str | None = None, endpoint: str | None = None) -> None:
        self.model = model
        self.endpoint = endpoint

    @abstractmethod
    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class NoneProvider(AnalysisProvider):
    name = "none"

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        questions = []
        for question in payload["questions"]:
            evidence = list(question["question_segment_ids"])
            for answer in question["answers"]:
                evidence.extend(answer["segment_ids"])
            questions.append(
                {
                    "question_id": question["question_id"],
                    "summary": "未启用语义分析模型；仅保留可追溯的结构化问答。",
                    "scores": {name: None for name in SCORE_NAMES},
                    "strengths": [],
                    "risks": [],
                    "suggestions": [],
                    "improved_answer": None,
                    "evidence_segment_ids": list(dict.fromkeys(evidence)),
                }
            )
        return {
            "overview": {
                "summary": "语义分析未启用；报告仅包含本地确定性结构结果。",
                "strengths": [],
                "risks": [],
                "suggested_focus": [],
            },
            "questions": questions,
            "review_plan": [],
            "warnings": ["No semantic model was enabled; scores and coaching were not generated."],
        }


SYSTEM_PROMPT = """You analyze technical interview answers. Return only one JSON object.
Do not invent facts. Every claim must be grounded in the provided question/answer text.
Use only provided question_id values. Evidence IDs must belong to the same question.
Score relevance, technical_depth, clarity, and structure from 1 to 5.
Return exactly these top-level keys: overview, questions, review_plan, warnings.
overview has summary, strengths, risks, suggested_focus (all lists except summary).
Each questions item has question_id, summary, scores, strengths, risks, suggestions,
improved_answer, evidence_segment_ids. review_plan items have priority, topic, action,
question_ids. Write coaching in the language used by the interview."""


class OpenAICompatibleProvider(AnalysisProvider):
    name = "openai-compatible"
    remote = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float = 120.0,
        connect_retries: int = 2,
    ) -> None:
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        super().__init__(model=model, endpoint=endpoint)
        self.remote = not _is_loopback_url(base_url)
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.connect_retries = connect_retries

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        httpx = _require_httpx()
        api_key = os.environ.get(self.api_key_env)
        if self.remote and not api_key:
            raise AnalysisError(f"环境变量 {self.api_key_env} 未设置")
        request_body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
        }
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            transport = httpx.HTTPTransport(retries=self.connect_retries)
            with httpx.Client(transport=transport, timeout=self.timeout_seconds) as client:
                response = client.post(self.endpoint, headers=headers, json=request_body)
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except Exception as exc:
            raise AnalysisError(f"OpenAI-compatible provider 请求失败: {exc}") from exc
        return _parse_json_content(str(content))


class OllamaProvider(AnalysisProvider):
    name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 180.0,
        connect_retries: int = 2,
    ) -> None:
        endpoint = f"{base_url.rstrip('/')}/api/chat"
        super().__init__(model=model, endpoint=endpoint)
        self.remote = not _is_loopback_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.connect_retries = connect_retries

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        httpx = _require_httpx()
        request_body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
        }
        try:
            transport = httpx.HTTPTransport(retries=self.connect_retries)
            with httpx.Client(transport=transport, timeout=self.timeout_seconds) as client:
                response = client.post(self.endpoint, json=request_body)
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except Exception as exc:
            raise AnalysisError(f"Ollama provider 请求失败: {exc}") from exc
        return _parse_json_content(str(content))


def create_provider(
    config: dict[str, Any],
    *,
    provider_name: str | None = None,
    model: str | None = None,
) -> AnalysisProvider:
    analysis_config = config.get("analysis", {})
    name = provider_name or str(analysis_config.get("provider", "none"))
    if name == "none":
        return NoneProvider()

    providers = analysis_config.get("providers", {})
    timeout = float(analysis_config.get("timeout_seconds", 120))
    connect_retries = int(analysis_config.get("connect_retries", 2))
    if timeout <= 0:
        raise AnalysisError("analysis.timeout_seconds 必须大于 0")
    if connect_retries < 0:
        raise AnalysisError("analysis.connect_retries 不能小于 0")
    if name == "openai-compatible":
        settings = providers.get("openai_compatible", {})
        selected_model = model or settings.get("model")
        if not selected_model:
            raise AnalysisError("openai-compatible provider 必须配置 model")
        base_url = str(settings.get("base_url", "https://api.openai.com/v1"))
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise AnalysisError("OpenAI-compatible base_url 必须是有效的 HTTP(S) URL")
        if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
            raise AnalysisError(
                "provider base_url 不能包含凭据、query 或 fragment；请使用环境变量提供 API key"
            )
        if parsed_url.scheme != "https" and not _is_loopback_url(base_url):
            raise AnalysisError("远程 OpenAI-compatible endpoint 必须使用 HTTPS")
        return OpenAICompatibleProvider(
            model=str(selected_model),
            base_url=base_url,
            api_key_env=str(settings.get("api_key_env", "INTERVIEW_RECAP_API_KEY")),
            timeout_seconds=timeout,
            connect_retries=connect_retries,
        )
    if name == "ollama":
        settings = providers.get("ollama", {})
        selected_model = model or settings.get("model")
        if not selected_model:
            raise AnalysisError("ollama provider 必须配置 model 或传入 --model")
        base_url = str(settings.get("base_url", "http://127.0.0.1:11434"))
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise AnalysisError("Ollama base_url 必须是有效的 HTTP(S) URL")
        if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
            raise AnalysisError("provider base_url 不能包含凭据、query 或 fragment")
        if not _is_loopback_url(base_url) and parsed_url.scheme != "https":
            raise AnalysisError("远程 Ollama endpoint 必须使用 HTTPS")
        return OllamaProvider(
            model=str(selected_model),
            base_url=base_url,
            timeout_seconds=timeout,
            connect_retries=connect_retries,
        )
    raise AnalysisError(f"未知 analysis provider: {name}")


def validate_provider_configuration(
    provider: AnalysisProvider,
    *,
    allow_remote: bool,
    redact: bool,
    require_http_client: bool = False,
) -> None:
    if provider.remote and not allow_remote:
        raise AnalysisError(
            f"provider {provider.name} 会发送脱敏后的问答文本；请显式传入 --allow-remote"
        )
    if provider.remote and not redact:
        raise AnalysisError("远程 provider 必须启用本地脱敏")
    if require_http_client and provider.name != "none":
        _require_httpx()


def _payload_batches(
    payload: dict[str, Any], *, maximum_questions: int, maximum_chars: int
) -> list[dict[str, Any]]:
    if maximum_questions < 1:
        raise AnalysisError("analysis.maximum_questions_per_request 必须大于 0")
    if maximum_chars < 1:
        raise AnalysisError("analysis.maximum_payload_chars 必须大于 0")
    questions = payload.get("questions") or []
    if not questions:
        return [payload]
    base = {key: value for key, value in payload.items() if key != "questions"}
    batches: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for question in questions:
        candidate = [*current, question]
        candidate_payload = {**base, "questions": candidate}
        candidate_size = len(
            json.dumps(
                candidate_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if current and (len(candidate) > maximum_questions or candidate_size > maximum_chars):
            batches.append({**base, "questions": current})
            current = [question]
        else:
            current = candidate
        single_size = len(
            json.dumps(
                {**base, "questions": current},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if len(current) == 1 and single_size > maximum_chars:
            raise AnalysisError(
                f"问题 {question['question_id']} 的最小化 payload 为 {single_size} 字符，"
                f"超过单次限制 {maximum_chars}；请降低 analysis.maximum_answer_chars"
            )
    if current:
        batches.append({**base, "questions": current})
    return batches


def _deduplicate_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _merge_batch_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) == 1:
        return results[0]
    summaries = [result["overview"]["summary"] for result in results]
    return {
        "overview": {
            "assessment_status": "evaluated",
            "summary": (
                f"已分 {len(results)} 批完成语义分析。"
                + " ".join(summary for summary in summaries if summary)
            ).strip(),
            "strengths": _deduplicate_strings(
                [item for result in results for item in result["overview"]["strengths"]]
            ),
            "risks": _deduplicate_strings(
                [item for result in results for item in result["overview"]["risks"]]
            ),
            "suggested_focus": _deduplicate_strings(
                [item for result in results for item in result["overview"]["suggested_focus"]]
            ),
        },
        "questions": [item for result in results for item in result["questions"]],
        "review_plan": [item for result in results for item in result["review_plan"]],
        "warnings": _deduplicate_strings(
            [item for result in results for item in result["warnings"]]
            + [f"Semantic analysis used {len(results)} requests."]
        ),
    }


def run_analysis(
    conversation: dict[str, Any],
    provider: AnalysisProvider,
    *,
    allow_remote: bool = False,
    redact: bool = True,
    custom_redaction_terms: list[str] | None = None,
    maximum_answer_chars: int = 2400,
    maximum_payload_chars: int = 160000,
    maximum_questions_per_request: int = 12,
) -> dict[str, Any]:
    if conversation.get("schema_version") != "0.2.0":
        raise AnalysisError("分析层只接受 conversation V2 JSON")
    validate_provider_configuration(
        provider, allow_remote=allow_remote, redact=redact, require_http_client=False
    )

    redactor = Redactor(custom_redaction_terms)
    payload = build_minimized_payload(
        conversation,
        redactor=redactor,
        redact=redact,
        maximum_answer_chars=maximum_answer_chars,
    )
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    batches = (
        [payload]
        if provider.name == "none"
        else _payload_batches(
            payload,
            maximum_questions=maximum_questions_per_request,
            maximum_chars=maximum_payload_chars,
        )
    )
    normalized_batches: list[dict[str, Any]] = []
    for batch in batches:
        expected = {item["question_id"] for item in batch["questions"]}
        raw_result = provider.analyze(batch)
        normalized_batches.append(
            normalize_provider_result(
                raw_result,
                conversation,
                evaluated=provider.name != "none",
                expected_question_ids=expected,
            )
        )
    normalized = _merge_batch_results(normalized_batches)
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_at": _now(),
        "source": {
            "conversation_schema_version": conversation.get("schema_version"),
            "question_count": len(payload["questions"]),
            "request_count": len(batches),
        },
        "provider": {
            "name": provider.name,
            "model": provider.model,
            "remote": provider.remote,
            "endpoint": provider.endpoint,
        },
        "privacy": {
            "remote_consent": bool(allow_remote) if provider.remote else False,
            "redaction_enabled": redact,
            "redaction_counts": redactor.audit_counts(),
            "sent_audio": False,
            "sent_full_transcript": False,
            "payload_scope": "question_tree_only",
            "request_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        },
        **normalized,
    }


def save_analysis(document: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
