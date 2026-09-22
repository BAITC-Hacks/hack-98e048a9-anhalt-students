# 🛠️ HackAlem AI — Curated Starter Repositories & Agent Skills

This document contains pre-selected templates, frameworks, and AI agent skills gathered for the **HackAlem AI** hackathon.

---

## 🎨 1. Frontend & UI Templates

| Project | Description | Quick Command |
|---|---|---|
| **[vercel/chatbot](https://github.com/vercel/chatbot)** | Next.js AI chatbot with streaming responses & Tailwind | `git clone https://github.com/vercel/chatbot.git hackalem-frontend` |
| **[shadcn/ui](https://github.com/shadcn-ui/ui)** | Beautifully designed, accessible React components | `npx shadcn@latest init` |
| **[create-t3-app](https://github.com/t3-oss/create-t3-app)** | Full-stack TypeScript app (Next.js, tRPC, Tailwind) | `npm create t3-app@latest` |
| **[open-design](https://github.com/nexu-io/open-design)** | Local-first design workspace for coding agents | Explore UI prototypes |

---

## ⚡ 2. Backend & Instant Databases

| Project | Description | Quick Command |
|---|---|---|
| **[pocketbase](https://github.com/pocketbase/pocketbase)** | Single-file backend with SQLite, Auth, REST & Realtime API | Download binary & run `./pocketbase serve` |
| **[fastapi-template](https://github.com/fastapi/full-stack-fastapi-template)** | Full-stack FastAPI + PostgreSQL + Docker | `git clone https://github.com/fastapi/full-stack-fastapi-template.git hackalem-backend` |
| **[supabase](https://github.com/supabase/supabase)** | Open-source Firebase alternative (PostgreSQL + Auth + Storage) | `npx supabase init` |

---

## 🤖 3. AI Agent Frameworks & RAG

| Project | Description | Quick Command |
|---|---|---|
| **[openai-agents-python](https://github.com/openai/openai-agents-python)** | Lightweight agent orchestration with handoffs & tool calling | `pip install openai-agents` |
| **[create-llama](https://github.com/run-llama/create-llama)** | Instant Agentic RAG generator (FastAPI / Next.js) | `npx create-llama@latest` |
| **[langgraph](https://github.com/langchain-ai/langgraph)** | Stateful, cyclical multi-agent workflows | `pip install langgraph` |
| **[crewAI-examples](https://github.com/crewAIInc/crewAI-examples)** | Role-playing autonomous AI agents | Reference implementations |

---

## 🧠 4. AI Agent Skills (Prompt & Behavior Enhancers)

| Skill / Repo | Purpose & Benefit |
|---|---|
| **[leonxlnx/taste-skill](https://github.com/leonxlnx/taste-skill)** | **Anti-slop UI design system**: Prevents AI from creating ugly generic interfaces; enforces clean typography, spacing, and modern palettes. |
| **[ayghri/i-have-adhd](https://github.com/ayghri/i-have-adhd)** | **Concise Agent Output**: Stops conversational fluff and forces the AI agent to give immediate next steps and ready-to-run code. |
| **[Panniantong/agent-reach](https://github.com/Panniantong/agent-reach)** | Web and social platform scraping/search for agents without expensive API keys. |
| **[virgiliojr94/book-to-skill](https://github.com/virgiliojr94/book-to-skill)** | Compresses documentation and PDFs into structured on-demand agent skills (20x token reduction). |
| **[scientific-agent-skills](https://github.com/k-dense-ai/scientific-agent-skills)** | Curated skills for scientific research, data processing, and analysis. |
| **[playwright agent-cli](https://playwright.dev/agent-cli/introduction)** | Headless browser automation for AI agents. |

---

> [!WARNING]
> **Avoid `unslothai/unsloth` during the 5-hour hackathon!** Model fine-tuning takes too long and introduces CUDA/dataset bottlenecks. Use Prompt Engineering, RAG, and Tool Calling instead.
