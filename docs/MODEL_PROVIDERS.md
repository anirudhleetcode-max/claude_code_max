# Model providers

The agent never talks to a provider API directly. It builds provider-neutral
requests (`ModelRequest`: messages of text / tool-use / tool-result blocks, tool
specs, limits) and receives `ModelResponse`s. Adapters in
`src/ai_engineer/providers/` translate both ways. No model id is hard-coded
anywhere: you choose them in configuration.

## Built-in providers

| Type | Transport | Tools | Structured output | Embeddings | Model listing | Verified against |
|---|---|---|---|---|---|---|
| `anthropic` | official `anthropic` SDK (extra) | native | prompt or native (`output_config.format`) | — | yes | mocked HTTP |
| `openai` | REST Chat Completions | native | prompt or native (`response_format`) | yes | yes | mocked HTTP |
| `openai_compatible` | same, custom `base_url` | native (or prompt) | prompt or native | yes | yes | mocked HTTP |
| `google` | Gemini REST `generateContent` | native | prompt or native (`responseSchema`) | yes | yes | mocked HTTP |
| `ollama` | native REST `/api/chat` | native (or prompt) | prompt or native (`format`) | yes | yes | mocked HTTP |
| `scripted` | none (deterministic transcript) | native | prompt | hashing (lexical) | yes | used by tests/benchmarks |

"Verified against mocked HTTP" means request construction and response parsing
are tested against the providers' documented formats, including streaming, error
mapping, tool calls and reasoning-block round trips. Live behaviour has not been
verified for every service; `aie providers test [role]` performs a real request.

Adapter details worth knowing:
- **Anthropic:** reasoning blocks are kept as opaque blocks and replayed unchanged
  to the same model; `tool_choice` is never forced; sampling parameters are only
  sent when configured; prompt caching is on by default; large requests stream;
  optional server-side refusal fallback (`options.server_side_fallback`).
- **Google:** raw response parts (including thought signatures) are replayed
  verbatim to the same model; tool schemas are converted to the OpenAPI subset
  Gemini accepts (refs inlined, unsupported keywords removed).
- **OpenAI / compatible:** `max_completion_tokens` vs `max_tokens` is
  configurable; tool results become `tool` messages directly after the call.
- **Ollama:** `num_ctx`, `top_p`, ... via `params`; works fully offline.

## Roles, fallback and failure handling

```toml
[models.roles]
default  = ["anthropic:<model>", "openai:<model>", "ollama:<model>"]
reviewer = ["openai:<model>", "anthropic:<model>"]
fast     = ["ollama:<model>"]
```

For each request the router (`models/router.py`):
1. skips models whose circuit breaker is open (after `failure_threshold` consecutive
   failures, for `reset_after_s`; then a trial call is allowed). Only failures that
   say the model is unhealthy count, such as exhausted retries, authentication or an
   unreachable endpoint. Request-specific errors do not: context length, invalid
   request, refusal and malformed output;
2. retries transient errors (timeouts, connection errors, 5xx, 429 honouring
   `retry-after`) with exponential backoff and jitter, `max_attempts` times;
3. moves to the next model on non-retryable errors (auth, invalid request, refusal,
   malformed structured output after repairs) and emits `MODEL_FALLBACK`;
4. raises `AllModelsFailedError` when the chain is exhausted — the task becomes
   `BLOCKED` with its state saved, and `aie resume` continues later.

Context-length errors are not "fallen back": the agent loop resets its
conversation with a progress summary and continues. Because conversations are
stored in the neutral format, switching providers mid-task needs no translation
of saved state.

Every call is timed; token usage is recorded when the provider reports it and
estimated (and flagged as estimated) otherwise.

## Models without native tool calling

Set `native_tools = false` for the model. Tools are then described in the system
prompt and the model answers with `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
blocks, which are parsed back into normal tool calls (malformed blocks are
reported to the model as errors).

## Adding a provider

1. Subclass `ModelProvider` (`src/ai_engineer/models/base.py`) and implement
   `async def _generate(self, req: ModelRequest) -> ModelResponse`. Optionally
   override `stream`, `embeddings`, `list_models`, `aclose`.
2. Translate `req.messages` (text, `ToolUseBlock`, `ToolResultBlock`; drop
   `OpaqueBlock`s that belong to other providers/models), `req.tools`
   (`name`, `description`, JSON-schema `input_schema`), `req.system`,
   `req.max_tokens`, and only send `temperature`/`stop` when set.
3. Map responses: text → `TextBlock`, tool calls → `ToolUseBlock` (parse arguments
   as JSON; on failure emit a `ToolUseBlock` named `invalid_tool_call`), stop
   reason → `StopReason`, usage → `Usage`.
4. Map errors to the taxonomy in `core/errors.py` (`RateLimitError`,
   `RetryableProviderError`, `ProviderUnavailableError`, `AuthenticationError`,
   `InvalidRequestError`, `ContextLengthError`). `providers/_http.py` has helpers
   for REST APIs.
5. Register it: `registry.register_type("myprovider", lambda name, cfg: MyProvider(name, cfg))`
   on `runtime.router.registry` (or add it to `_builtin_factories` in
   `models/registry.py`), then configure `[models.providers.x] type = "myprovider"`.
6. Test with `httpx.MockTransport` like `tests/unit/test_providers.py`, then
   `aie providers test`.
