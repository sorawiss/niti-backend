# One-turn RAG Chat API

`POST /chat` connects the existing Qdrant retrieval flow to the LLM gateway.
It is stateless: the backend does not create conversations or store messages.

## Request flow

```text
frontend question
  -> retrieve_laws(question)
  -> assign LAW_1..LAW_N source IDs
  -> build a legal-context prompt
  -> OpenRouter structured output
  -> server-side citation validation
  -> answer and retrieved citations
```

The model never supplies citation text shown to the user. It only selects
source IDs; citation text comes from Qdrant retrieval results.

## API request

```http
POST /chat
Content-Type: application/json

{
  "message": "นายจ้างต้องจ่ายค่าล่วงเวลาอย่างไร",
  "top_k": 5,
  "law_name": null
}
```

`message` is required. `top_k` accepts 1-10. `law_name` is an optional exact
metadata filter.

## API response

```json
{
  "status": "answered",
  "answer": "...",
  "citations": [
    {
      "source_id": "LAW_1",
      "score": 0.91,
      "law_name": "พระราชบัญญัติคุ้มครองแรงงาน",
      "section": "61",
      "text": "...",
      "source_file": "chap2.jsonl"
    }
  ],
  "requires_human_review": false,
  "model": "openai/gpt-5-mini",
  "provider": "Azure",
  "usage": {
    "prompt_tokens": 500,
    "completion_tokens": 120,
    "total_tokens": 620,
    "cost": 0.001
  }
}
```

Possible statuses are `answered` and `insufficient_context`. If retrieval finds
nothing, the workflow skips the paid LLM call.

## Frontend example

```ts
type ChatResponse = {
  status: "answered" | "insufficient_context";
  answer: string;
  citations: Array<{
    source_id: string;
    score: number | null;
    law_name: string;
    section: string;
    text: string;
    source_file: string | null;
  }>;
  requires_human_review: boolean;
  model: string | null;
  provider: string | null;
};

export async function askLegalQuestion(message: string): Promise<ChatResponse> {
  const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, top_k: 5 }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail?.message ?? "Chat request failed");
  }

  return response.json();
}
```

## Run locally

```powershell
.\.venv\Scripts\uvicorn.exe app.main:app --reload
```

Open `http://localhost:8000/docs` to test `/chat` using FastAPI Swagger UI.

## Error contract

Errors use FastAPI's `detail` envelope:

```json
{
  "detail": {
    "code": "llm_rate_limit",
    "message": "บริการ AI ไม่พร้อมใช้งาน กรุณาลองใหม่ภายหลัง",
    "retryable": true
  }
}
```

The endpoint returns 503 for retryable model failures, 502 for invalid model
responses, 422 for invalid request data, and 500 for retrieval/workflow errors.

## Security behavior

- The prompt instructs the model to use only retrieved law sources.
- Retrieved chunks and user input are treated as data, not instructions.
- OpenRouter structured output is validated with Pydantic.
- Unknown or invented citation IDs are removed.
- An `answered` response with no valid citation is not exposed as an answer.
- ZDR and provider fallback behavior come from the LLM gateway configuration.

## Official references

- FastAPI request bodies:
  https://fastapi.tiangolo.com/tutorial/body/
- FastAPI dependencies:
  https://fastapi.tiangolo.com/tutorial/dependencies/
- FastAPI lifespan events:
  https://fastapi.tiangolo.com/advanced/events/
- OpenRouter structured outputs:
  https://openrouter.ai/docs/guides/features/structured-outputs
