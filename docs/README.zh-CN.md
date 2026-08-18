# Interview Recap 中文说明

Interview Recap 是一个隐私优先的本地面试录音转写、对话结构修复与可选语义分析工具。

音频预处理、ASR、VAD、标点、说话人分离、segment 清洗、角色修复、阶段识别和问题树提取均在本地运行。语义分析位于最后一层，默认使用 `none` provider，不调用 LLM，也不会上传面试数据。

## 输出内容

- 完整保留的 FunASR raw JSON；
- 带时间戳和 diarization speaker ID 的转写；
- 每段独立的 `speaker_id`、`role`、`role_confidence` 和 `phase`；
- 不删除源 segment 的派生清洗结果；
- coding 静音、低语音密度和噪声事件；
- `Q1`、`Q1.1`、`Q1.2` 形式的可追溯 Question Tree；
- 每次面试独立的 session workspace、`manifest.json` 和断点续跑；
- 可选的本地 Ollama 或 OpenAI-compatible 语义分析；
- 供程序使用的 JSON 和供人工阅读的 Markdown。

V2 结构修复是确定性规则，不依赖 Codex 或其他 LLM。Provider-neutral 分析层只消费 V2 Question Tree，模型返回结构化 analysis JSON，最终报告仍由本地确定性 renderer 生成。

## 安装

需要 Python 3.10–3.12、FFmpeg 和 FFprobe。

macOS：

```bash
brew install ffmpeg
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[asr]"
```

Ubuntu/Debian：

```bash
sudo apt-get update
sudo apt-get install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[asr]"
```

检查环境：

```bash
interview-recap env-check --format text
```

第一次 ASR 运行可能从配置的模型仓库下载数 GiB 模型。后续会优先使用完整的本地 ModelScope snapshot。

CUDA 用户应先按照 PyTorch 官方说明安装匹配的 Torch/Torchaudio，再安装 `.[asr]`。项目使用兼容版本范围，不强制所有平台使用同一个 Torch wheel。如果 Python 3.12 构建时出现旧 `pkg_resources` 或 `ImpImporter` 错误，请重新创建虚拟环境并按上面的命令升级 `setuptools`。

## 快速开始

将 `.m4a`、`.mp3` 或 `.wav` 放入 `input/`：

```bash
interview-recap run input/interview.m4a
```

完整运行会创建稳定的私有会话目录：

```text
work/sessions/<session-id>/
  manifest.json
  audio/interview.16k.wav
  raw/interview.funasr.json
  structured/interview.json
  structured/interview.v2.json
  analysis/analysis.json
  reports/report.md
```

中途中断后可以跳过已经完成且产物仍存在的阶段：

```bash
interview-recap run input/interview.m4a --resume
```

使用 `--overwrite` 才会明确重跑所有阶段。进度写入 stderr，最终路径 JSON 写入 stdout。这些目录默认不会进入 Git。

## 分阶段执行

```bash
interview-recap preprocess input/interview.m4a
interview-recap transcribe work/audio/interview.16k.wav
interview-recap structure transcripts/raw/interview.funasr.json
interview-recap segment transcripts/structured/interview.json
interview-recap repair-v2 transcripts/structured/interview.json
interview-recap analyze transcripts/structured/interview.v2.json --provider none
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

## Provider-neutral 分析层

默认 `none` provider 不调用模型，只生成带问题、回答、时间戳和 segment ID 的确定性报告。

使用 Ollama 或远程兼容接口时安装可选依赖：

```bash
python -m pip install -e ".[analysis]"
```

本地 Ollama：

```bash
interview-recap analyze path/to/interview.v2.json \
  --provider ollama \
  --model qwen2.5:7b
```

OpenAI-compatible 接口：

```bash
export INTERVIEW_RECAP_API_KEY="..."
interview-recap analyze path/to/interview.v2.json \
  --provider openai-compatible \
  --model your-model \
  --allow-remote
```

远程调用必须显式提供 `--allow-remote`。发送前只提取 Question Tree，并在本地脱敏邮箱、手机号、身份证号、IP、凭据模式以及自定义敏感词。不会发送音频、完整 segments、说话人 ID 或源文件路径。

公司名、姓名等项目特定词汇应逐行写入被 Git 忽略的本地文件，例如 `work/redaction_terms.txt`，再将 `analysis.redaction_terms_file` 指向它。不要把真实敏感词直接写入会被提交的 `config.yaml`。

模型输出必须通过 [analysis-v1.schema.json](../schemas/analysis-v1.schema.json) 对应的约束：每个问题都要使用真实 `question_id`，证据只能引用该问答原有的 `segment_ids`。缺题、未知 ID、越界评分或伪造证据都会被拒绝。

长面试会按 `analysis.maximum_questions_per_request` 分批分析，并按原顺序合并。`maximum_answer_chars` 与 `maximum_payload_chars` 限制每次请求的文本规模，避免一次请求承载整场长面试。

## 隐私模式

默认配置完全不调用远程 LLM。只有用户选择远程 provider 并传入 `--allow-remote` 后，脱敏的 Question Tree 文本才会离开本机。录音、转写、分析与报告默认被 `.gitignore` 排除。请勿在公开 issue 或 PR 中粘贴真实面试文本。

第三方 provider 的留存、训练与合规政策不属于本项目保证范围。详细威胁模型见 [privacy.md](privacy.md)。

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
- LLM 评分属于模型判断，不是真值，应结合 evidence segment IDs 人工复核；
- 合成音频生成目前依赖 macOS `say`。

## 许可证

项目源码使用 [Apache License 2.0](../LICENSE)。下载的模型权重和第三方依赖遵循各自上游许可证，本仓库不重新分发模型权重。上游项目与模型清单见 [THIRD_PARTY.md](../THIRD_PARTY.md)。
