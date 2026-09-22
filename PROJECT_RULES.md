# 🤖 PROJECT RULES & INSTRUCTIONS FOR AI AGENTS

> **Context:** HackAlem AI Hackathon (Astana Hub, 5-hour sprint).  
> **Audience:** Technical Experts, Jury, and Automated AI Judge.  
> **Core Objective:** Build a stable, working MVP with clean code and a reproducible single-command run script.

---

## 🎯 1. Behavioral & Execution Rules (CRITICAL)

1. **No Code Truncation / No Laziness:**
   - NEVER use `// TODO: implement later` or `// ... rest of code remains the same ...`.
   - ALWAYS output complete, working, syntactically valid code blocks.
2. **Concise & Direct Action (ADHD-Style):**
   - Skip conversational fluff, generic intros, and redundant explanations.
   - Provide the immediate command or file edit first, followed by a 1-line explanation.
3. **Environment Variable Hygiene:**
   - Whenever introducing a new environment variable in code, IMMEDIATELY add it with a dummy or default value to `.env.example`.
4. **Mock / Fallback Mode is Mandatory (Rule 5.6.6 Compliance):**
   - Judges must be able to test the app without inputting their personal paid API keys.
   - If an API key is missing or fails, gracefully fall back to realistic mock responses (`DEMO_MOCK_MODE=true`).

---

## 🎨 2. UI/UX Design Standards ("Anti-Slop" / Taste-Skill)

1. **Modern, Minimalist Palette:**
   - Use Tailwind CSS and `shadcn/ui` conventions (slate/zinc neutral tones, clean borders, subtle shadows).
   - Avoid generic rainbow gradients or cartoonish AI styling.
2. **Typography & Spacing:**
   - Use clean sans-serif typography (Inter or Geist).
   - Maintain consistent padding (`p-4`, `p-6`), clear hierarchy (`h1`, `h2`, `muted-foreground`).
3. **Feedback & States:**
   - Always include loading spinners (`lucide-react/Loader2`), empty states, and error toasts.

---

## 🏗️ 3. Architecture & Code Quality

1. **Modular Architecture:**
   - `components/` — reusable UI widgets.
   - `services/` or `lib/` — API clients, LLM orchestration, and business logic.
   - `types/` — TypeScript interfaces or Python Pydantic models.
2. **Validation & Error Handling:**
   - Always wrap async requests in `try / catch` blocks.
   - Return clean HTTP status codes (200, 400, 422, 500) with descriptive error messages.
3. **Automated Verification:**
   - Every core function must have a simple verification script or unit test to guarantee it works.

---

## ⏱️ 4. Hackathon Time & Git Protocol

- **Every 60 Minutes:** Remind the team or initiate an incremental Git commit with a descriptive message (e.g., `feat(api): add streaming response endpoint - Hour 2 checkpoint`).
- **Never Break the Main Branch:** All commits on `main` must be buildable and runnable.
