from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from .transcript import format_timestamp


def _flatten_questions(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for node in nodes:
        flattened.append(node)
        flattened.extend(_flatten_questions(node.get("children") or []))
    return flattened


def _safe_text(value: Any) -> str:
    escaped = html.escape(str(value), quote=False)
    escaped = escaped.replace("\\", "\\\\")
    escaped = re.sub(r"([`*_\[\]])", r"\\\1", escaped)
    return escaped.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def _inline_code(value: Any) -> str:
    escaped = html.escape(str(value), quote=False)
    return escaped.replace("`", "&#96;").replace("\r", " ").replace("\n", " ")


def _add_list(lines: list[str], heading: str, values: list[str]) -> None:
    if not values:
        return
    lines.extend([f"**{heading}**", ""])
    lines.extend(f"- {_safe_text(value)}" for value in values)
    lines.append("")


def render_analysis_report(conversation: dict[str, Any], analysis: dict[str, Any]) -> str:
    provider = analysis.get("provider") or {}
    privacy = analysis.get("privacy") or {}
    statistics = conversation.get("statistics") or {}
    overview = analysis.get("overview") or {}
    lines = [
        "# Interview Recap",
        "",
        f"- Provider：`{_inline_code(provider.get('name', 'unknown'))}`",
        f"- Model：`{_inline_code(provider.get('model') or 'none')}`",
        f"- 远程传输：{'是（已显式授权）' if provider.get('remote') else '否'}",
        f"- 分析状态：`{_inline_code(overview.get('assessment_status', 'unknown'))}`",
        "",
        "## 对话结构摘要",
        "",
        f"- 有效 segments：{statistics.get('valid_segment_count', 0)}",
        f"- 主问题：{statistics.get('main_question_count', 0)}",
        f"- 追问：{statistics.get('follow_up_count', 0)}",
        f"- 澄清：{statistics.get('clarification_count', 0)}",
        f"- Coding events：{statistics.get('coding_event_count', 0)}",
        f"- Unknown role：{statistics.get('unknown_role_count', 0)}",
        "",
        "## 总体分析",
        "",
        _safe_text(overview.get("summary") or "暂无总体分析。"),
        "",
    ]
    _add_list(lines, "优势", overview.get("strengths") or [])
    _add_list(lines, "风险", overview.get("risks") or [])
    _add_list(lines, "建议重点", overview.get("suggested_focus") or [])

    analysis_by_id = {item["question_id"]: item for item in analysis.get("questions") or []}
    lines.extend(["## 问题与回答", ""])
    for node in _flatten_questions(conversation.get("question_tree") or []):
        item = analysis_by_id.get(node["id"], {})
        stamp = f"{format_timestamp(node['start'])}–{format_timestamp(node['end'])}"
        topic = (node.get("topic") or {}).get("label", "未分类")
        lines.extend(
            [
                f"### {_safe_text(node['id'])} · {_safe_text(topic)}",
                "",
                f"- 类型：`{_inline_code(node.get('node_type', 'unknown'))}`",
                f"- 阶段：`{_inline_code(node.get('phase', 'UNKNOWN'))}`",
                f"- 时间：`{stamp}`",
                f"- 问题 segments：`{_inline_code(', '.join(node.get('segment_ids') or []))}`",
                "",
                f"**问题**：{_safe_text(node.get('text', ''))}",
                "",
            ]
        )
        answers = node.get("answers") or []
        if answers:
            lines.extend(["**回答**", ""])
            for answer in answers:
                answer_stamp = (
                    f"{format_timestamp(answer['start'])}–{format_timestamp(answer['end'])}"
                )
                ids = ", ".join(answer.get("segment_ids") or [])
                lines.append(
                    f"- [{answer_stamp}; {_safe_text(ids)}] {_safe_text(answer.get('text', ''))}"
                )
            lines.append("")
        else:
            lines.extend(["**回答**：未检测到可关联回答。", ""])

        scores = item.get("scores") or {}
        if any(value is not None for value in scores.values()):
            labels = {
                "relevance": "相关性",
                "technical_depth": "技术深度",
                "clarity": "清晰度",
                "structure": "结构",
            }
            score_text = "；".join(
                f"{labels.get(name, name)} {score}/5"
                for name, score in scores.items()
                if score is not None
            )
            lines.extend([f"**评分**：{score_text}", ""])
        if item.get("summary"):
            lines.extend([f"**分析**：{_safe_text(item['summary'])}", ""])
        _add_list(lines, "回答优势", item.get("strengths") or [])
        _add_list(lines, "风险与缺口", item.get("risks") or [])
        _add_list(lines, "改进建议", item.get("suggestions") or [])
        if item.get("improved_answer"):
            lines.extend(["**改进版回答**", "", _safe_text(item["improved_answer"]), ""])
        evidence = item.get("evidence_segment_ids") or []
        if evidence:
            lines.extend([f"证据 segments：`{_inline_code(', '.join(evidence))}`", ""])

    plan = analysis.get("review_plan") or []
    if plan:
        lines.extend(["## 复习计划", ""])
        for item in plan:
            question_ids = ", ".join(item.get("question_ids") or [])
            lines.append(
                f"- **{_safe_text(item.get('priority', 'medium'))} · "
                f"{_safe_text(item.get('topic', ''))}**："
                f"{_safe_text(item.get('action', ''))}（{_safe_text(question_ids)}）"
            )
        lines.append("")

    lines.extend(
        [
            "## 隐私与追溯",
            "",
            f"- 发送音频：{privacy.get('sent_audio', False)}",
            f"- 发送完整 transcript：{privacy.get('sent_full_transcript', False)}",
            f"- Provider payload：`{_inline_code(privacy.get('payload_scope', 'unknown'))}`",
            f"- 本地脱敏：{privacy.get('redaction_enabled', False)}",
            f"- 脱敏计数：`{_inline_code(privacy.get('redaction_counts', {}))}`",
            f"- Request SHA-256：`{_inline_code(privacy.get('request_sha256', ''))}`",
            "",
        ]
    )
    _add_list(lines, "Warnings", analysis.get("warnings") or [])
    return "\n".join(lines).rstrip() + "\n"


def save_analysis_report(
    conversation: dict[str, Any], analysis: dict[str, Any], output_path: str | Path
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_analysis_report(conversation, analysis), encoding="utf-8")
    return path
