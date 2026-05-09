# AI Personal Knowledge Base / AI 个人知识库

本地优先的 AI 研究助手。不仅仅是"对话你的文档"——它会记住你学过什么、追踪学习进度、发现知识盲区。

核心思路：把 RAG（检索增强生成）和**学习追踪**结合起来。每次问答都会被记录，系统自动计算每个主题的掌握度，告诉你哪些文档还没读过，哪些知识领域还是空白。

## 功能

### 基础问答
基于你的文档库（PDF/TXT/MD）进行检索增强生成，回答带 `[1]` `[2]` 来源引用，可追溯每条信息的出处。

### 智能体研究模式
开启后自动将复杂问题拆解为 2-4 个子问题，逐一检索并综合成完整回答。可选启用 DuckDuckGo 网络搜索作为补充。

### 多后端 LLM 支持
| 后端 | 说明 |
|------|------|
| Ollama | 本地运行，免费，无需联网 |
| vLLM | 自建或内网 OpenAI 兼容端点 |
| OpenAI | 任何 OpenAI 兼容 API |
| Anthropic | Claude 系列模型 |

在应用内点击"设置"即可切换，无需修改代码。

### 学习仪表盘
右侧面板实时展示：
- **主题** — 各学习主题及掌握度评分 (1-5)，进度条颜色从红→黄→绿渐变
- **动态** — 当前会话的问答历史
- **知识缺口** — 已添加文档但尚未提问学习的主题，用琥珀色卡片标出

### 其他
- Markdown 导出：按主题导出完整学习笔记
- 文档管理：支持上传 PDF/TXT/MD，自动分块索引
- 环境变量配置：所有参数可通过 `KB_*` 环境变量覆盖

## 部署

### 环境要求
- Python 3.10+
- 8GB+ RAM（推荐 16GB）
- （可选）[Ollama](https://ollama.com) 用于本地模型

### 1. 克隆项目
```bash
git clone https://github.com/Hjh-12138/Personal-Knowledge-Base.git
cd Personal-Knowledge-Base
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

核心依赖：`langchain` `chromadb` `sentence-transformers` `customtkinter` `duckduckgo-search`

### 3. （可选）拉取本地模型
```bash
ollama pull qwen2.5:7b
```

如果使用 OpenAI / Anthropic 等云端 API，跳过此步，在应用内配置 API Key 即可。

### 环境变量（可选）
```bash
export KB_MODEL_BACKEND=openai      # ollama | vllm | openai | anthropic
export KB_MODEL_NAME=gpt-4o-mini
export KB_API_KEY=sk-xxx
export KB_DEBUG=true
```

## 使用说明

### 桌面应用（推荐）
```bash
python desktop_app.py
```

1. 点击 **上传文档** 选择 PDF/TXT/MD 文件，或将文件放入 `./documents/` 目录后点击 **重新扫描**
2. 在底部输入框输入问题，按回车发送
3. 勾选 **智能体模式** 处理复杂问题；勾选 **网络搜索** 启用联网补充
4. 右侧仪表盘查看学习进度和知识缺口
5. 顶栏 **设置** 切换 LLM 后端；**导出** 保存当前主题笔记

### Web 界面（备选）
```bash
streamlit run streamlit_app.py
```
浏览器打开 `http://localhost:8501`，左侧对话右侧仪表盘的布局。

### CLI
```bash
python -m knowledge_base stats                    # 学习统计
python -m knowledge_base topics                   # 主题列表
python -m knowledge_base export --topic "物理学"   # 导出 Markdown
python -m knowledge_base search "什么是瑞利散射"    # 检索知识库
```

## 项目结构
```
knowledge_base/          # 核心库
├── config.py            # 配置项（支持 KB_* 环境变量）
├── llm.py               # LLM 工厂（ollama/vllm/openai/anthropic）
├── ingestion.py         # 文档加载、分块、嵌入、Chroma 存储
├── retrieval.py         # 向量检索 + 嵌入模型缓存
├── generation.py        # RAG 提示词 + 生成
├── agent.py             # 智能体：问题拆解、联网搜索、结果综合
├── tracker.py           # SQLite 学习追踪（WAL + SAVEPOINT）
├── embedding.py         # 嵌入函数封装
└── __main__.py          # CLI 入口

desktop_app.py           # 桌面 GUI（CustomTkinter，中文界面）
streamlit_app.py         # Web GUI（Streamlit）

tests/                   # 59 个测试
```

## 配置参考
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_backend` | ollama | LLM 后端类型 |
| `model_name` | qwen2.5:7b | 模型名称 |
| `chunk_size` | 1000 | 文档分块大小（字符） |
| `top_k` | 4 | 每次检索返回的文档数 |
| `temperature` | 0.3 | 生成温度 (0-1) |
| `timeout` | 60 | LLM 超时（秒） |
| `debug` | false | 调试日志 |

## 测试
```bash
pytest tests/ -v
```
59 个测试，1 个跳过（langchain-anthropic 可选依赖）。测试不需要运行 Ollama。

## 隐私
所有数据本地存储：文档、嵌入向量、问答记录、学习数据均不离开你的机器。

## License
MIT
