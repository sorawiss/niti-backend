# LLM Gateway: Phase 1

This phase adds a provider-neutral boundary between application logic and
OpenRouter. It does not expose a chat endpoint yet; that belongs to Phase 2.

## Why the gateway exists

Application code should depend on `LLMGateway`, not OpenRouter request or
response objects. The current dependency direction is:

```text
future ChatService -> LLMGateway <- OpenRouterGateway -> OpenRouter API
```

Changing providers later means adding another `LLMGateway` implementation.
The ChatService contract remains unchanged.

## Files

- `app/llm/base.py`: provider-neutral abstract interface.
- `app/llm/models.py`: request, response, message, and usage contracts.
- `app/llm/config.py`: environment configuration; secrets use `SecretStr`.
- `app/llm/openrouter.py`: OpenRouter HTTP adapter.
- `app/llm/errors.py`: errors that the application can handle consistently.
- `app/llm/factory.py`: provider construction in one location.
- `tests/unit/test_openrouter_gateway.py`: network-free contract tests.

## Configure it

Copy the LLM variables from `.env.example` into your ignored `.env` file.
Never commit the real API key.

Important defaults:

```env
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=openai/gpt-5-mini
LLM_ZERO_DATA_RETENTION=true
LLM_ALLOW_FALLBACKS=false
```

`zdr=true` restricts routing to Zero Data Retention endpoints.
`allow_fallbacks=false` prevents OpenRouter from silently trying a different
provider endpoint when the preferred route is unavailable. Model identity is
still returned and should be recorded with every future chat response.

## Use it from Python

```python
import asyncio

from app.llm import ChatMessage, LLMRequest, create_llm_gateway
from app.llm.models import ChatRole


async def main() -> None:
    gateway = create_llm_gateway()
    try:
        response = await gateway.generate(
            LLMRequest(
                messages=[
                    ChatMessage(
                        role=ChatRole.USER,
                        content="อธิบายสิทธิค่าล่วงเวลาแบบสั้น ๆ",
                    )
                ]
            )
        )
        print(response.content)
        print(response.model, response.provider, response.usage)
    finally:
        await gateway.close()


asyncio.run(main())
```

The API key is loaded only while the gateway is being created. Pydantic's
`SecretStr` prevents accidental disclosure through normal model display, but
application logs must still never log HTTP authorization headers.

## Request flow

1. `LLMRequest` validates messages and optional generation settings.
2. `OpenRouterGateway` selects request overrides or environment defaults.
3. It sends `POST /api/v1/chat/completions` with Bearer authentication.
4. The adapter maps the provider response into `LLMResponse`.
5. HTTP and transport failures become provider-neutral `LLMGatewayError`
   subclasses with a `retryable` flag.

The adapter does not automatically retry. A later service layer can retry only
idempotent, retryable failures with a bounded backoff policy.

## Structured output preparation

Phase 1 accepts an optional `response_format`. When supplied, the adapter also
sets `provider.require_parameters=true`, so OpenRouter will only select an
endpoint that supports the requested parameter. Phase 2 can pass a strict JSON
Schema for legal answers and validate the returned JSON with Pydantic.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use `httpx.MockTransport`: no request reaches OpenRouter, the real API
key is not read, and no credits are consumed.

## Official references

- OpenRouter chat completion API:
  https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
- OpenRouter authentication and quickstart:
  https://openrouter.ai/docs/quickstart
- Provider routing and `allow_fallbacks`:
  https://openrouter.ai/docs/guides/routing/provider-selection
- Zero Data Retention:
  https://openrouter.ai/docs/guides/features/zdr
- Structured outputs:
  https://openrouter.ai/docs/guides/features/structured-outputs
- HTTPX async client:
  https://www.python-httpx.org/async/
- HTTPX mock transport:
  https://www.python-httpx.org/advanced/transports/#mock-transports
- Pydantic `SecretStr`:
  https://docs.pydantic.dev/latest/api/types/#pydantic.types.SecretStr
