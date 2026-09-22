# 📋 Project Specification Template (Fill at 13:00)

> Spend 15–20 minutes to fill this document immediately after the track announcement. Once filled, provide this file to the AI Agent to kick off development.

---

## 1. Track & Problem Statement
* **Selected Track:** [e.g., Track 3: AI Customer Assistant / Healthcare / GovTech]
* **Target Users:** [Who is this for? e.g., Students, Call-center managers, Doctors]
* **Core Pain Point:** [What exact problem does this solve in 1 sentence?]
* **Value Proposition:** [Why is our solution 10x better than existing alternatives?]

---

## 2. Solution Overview & Key Features
* **Feature 1 (MVP Must-Have):** [e.g., File upload + semantic search via embeddings]
* **Feature 2 (MVP Must-Have):** [e.g., AI Agent with 3 custom tools for data extraction]
* **Feature 3 (Demo Polish):** [e.g., Interactive analytics dashboard with export to PDF]

---

## 3. User Journey (Demo Script for Jury)
1. **Step 1:** User logs in / lands on homepage and sees [X].
2. **Step 2:** User inputs [Prompt / File / Action].
3. **Step 3:** System executes workflow, agent calls tool [Y].
4. **Step 4:** User gets instant structured result with visualization [Z].

---

## 4. Technical Contracts & Schemas
* **Input Schema:**
  ```json
  {
    "query": "string",
    "filters": {}
  }
  ```
* **Output Schema:**
  ```json
  {
    "status": "success",
    "summary": "string",
    "data_points": []
  }
  ```

---

## 5. Tools & Third-Party Services
* **LLM Provider:** [Google Gemini / OpenAI]
* **Database:** [PocketBase / SQLite / Supabase]
* **External APIs / Scrapers:** [None / Custom REST API]
