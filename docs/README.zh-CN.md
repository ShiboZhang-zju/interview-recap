# Interview Recap 中文说明

Interview Recap 是一个隐私优先的本地面试录音转写与对话结构修复工具。

音频预处理、ASR、VAD、标点、说话人分离、segment 清洗、角色修复、阶段识别和问题树提取均在本地运行。核心 pipeline 不调用 LLM，也不会上传面试数据。

## 输出内容

- 完整保留的 FunASR raw JSON；
- 带时间戳和 diarization speaker ID 的转写；
- 每段独立的 `speaker_id`、`role`、`role_confidence` 和 `phase`；
- 不删除源 segment 的派生清洗结果；
- coding 静音、低语音密度和噪声事件；
- `Q1`、`Q1.1`、`Q1.2` 形式的可追溯 Question Tree；
- 供程序使用的 JSON 和供人工阅读的 Markdown。

V2 结构修复是确定性规则，不依赖 Codex 或其他 LLM。后续可以增加 provider-neutral 的可选 LLM 分析层，但它不会成为核心流程的必需依赖。

## 安装

需要 Python 3.10–3.12、FFmpeg 和 FFprobe。

macOS：

```bash
brew install ffmpeg
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[asr]"
```

Ubuntu/Debian：

```bash
sudo apt-get update
sudo apt-get install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[asr]"
```

检查环境：

```bash
interview-recap env-check
```

第一次 ASR 运行可能从配置的模型仓库下载数 GiB 模型。后续会优先使用完整的本地 ModelScope snapshot。

## 快速开始

将 `.m4a`、`.mp3` 或 `.wav` 放入 `input/`：

```bash
interview-recap run input/interview.m4a
```

输出目录：

```text
work/audio/               预处理 WAV
transcripts/raw/          原始 FunASR 输出
transcripts/structured/   V1 和 V2 JSON/Markdown
```

这些目录默认不会进入 Git。

## 分阶段执行

```bash
interview-recap preprocess input/interview.m4a
interview-recap transcribe work/audio/interview.16k.wav
interview-recap structure transcripts/raw/interview.funasr.json
interview-recap segment transcripts/structured/interview.json
interview-recap repair-v2 transcripts/structured/interview.json
```

`repair-v2` 只生成 `.v2.json` 和 `.v2.md`，不会覆盖 raw 或 V1 文件。

## V2 结构修复

### Segment Cleanup

每个源 segment 都会保留，并新增 `cleanup`。检测包括纯标点、重复噪声、长 duration + 极低文本密度，以及长 duration + 极少文本。短追问不会仅因字数少被删除。

### Role Repair

- `speaker_id` 来自 diarization；
- `role` 仅为 `interviewer`、`candidate`、`unknown`；
- `role_confidence` 按 segment 推断。

说话人标签只作为弱证据。当前文本、问题词法、前后文、phase 和 conversation state 可以纠正 speaker drift。

### Interview Phase

支持：

```text
INTRO
PROJECT
TECH_QA
CODING
CODING_DISCUSSION
BEHAVIORAL
CANDIDATE_QUESTIONS
END
UNKNOWN
```

coding 长静音和异常噪声会保存为 event，不会自动生成 Q&A；真实口头追问仍然保留。

### Question Tree

问题节点分为 `main_question`、`follow_up` 和 `clarification`，根节点还会标记 `topic_switch`。所有问题和回答均保留源 `segment_ids` 与时间戳。

公开的合成示例见 [examples/synthetic_interview.v2.json](../examples/synthetic_interview.v2.json)，JSON 约束见 [schemas/conversation-v2.schema.json](../schemas/conversation-v2.schema.json)。

## 隐私模式

当前版本完全不调用远程 LLM。录音、转写与报告默认被 `.gitignore` 排除。请勿在公开 issue 或 PR 中粘贴真实面试文本。

未来加入远程 LLM 时，应保持默认关闭、显式授权、发送最小化、先本地脱敏。详细威胁模型见 [privacy.md](privacy.md)。

## 测试

```bash
python -m unittest discover -s tests -v
```

单元测试不下载模型、不处理真实录音。macOS 可额外生成合成 smoke 音频：

```bash
interview-recap make-smoke-audio
```

## 已知限制

- ASR 结果不能作为逐字引用稿，精确措辞需要回听原音；
- 罕见中英文术语可能需要热词与人工纠正；
- 重叠说话和极短插话仍可能影响 diarization；
- V2 是确定性启发式规则，复杂反问和多位面试官场景需要人工检查；
- 合成音频生成目前依赖 macOS `say`。

## 许可证

项目源码使用 [Apache License 2.0](../LICENSE)。下载的模型权重和第三方依赖遵循各自上游许可证，本仓库不重新分发模型权重。上游项目与模型清单见 [THIRD_PARTY.md](../THIRD_PARTY.md)。
