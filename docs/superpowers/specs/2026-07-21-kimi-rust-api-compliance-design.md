---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Kimi API Compliance in the Rust Frontend
subtitle: Apply the existing Kimi request policy independently of the chat processor
---

The Kimi API compliance flags introduced by
[Inferact/dynamo pull request 85](https://github.com/Inferact/dynamo/pull/85) currently affect only
the vLLM Python chat processor. The default Rust chat processor accepts the flags but does not apply
their request defaults or allowlists. This design moves the same explicit, opt-in policy to the Rust
OpenAI ingress so both processors expose the same Kimi API contract.

## Goals

- Preserve pull request 85 as the source of the Kimi compliance contract.
- Enforce that contract when `--dyn-chat-processor=dynamo` or
  `--dyn-chat-processor=vllm` is selected.
- Keep compliance explicitly controlled by `--kimi-api-compliance`; do not infer it from the model
  name.
- Default the effective reasoning effort to `max` when thinking is enabled and the request omits an
  effort.
- Return OpenAI-compatible HTTP 400 responses for rejected Kimi parameters.
- Preserve the existing Rust K3 chat template, structural-tag behavior, tool parser, and reasoning
  parser.

## Non-Goals

- Do not replace the Rust chat processor with the vLLM processor.
- Do not change K3 tool-call or reasoning output parsing.
- Do not add a second Kimi-specific set of CLI flags or model-name detection.
- Do not change behavior when `--kimi-api-compliance` is disabled.
- Do not make `max_completion_tokens` a hard cap. The configured value remains an omission default.
- Do not extend `/v1/responses`. Pull request 85 applies to Chat Completions, and the current
  Responses protocol cannot represent the Kimi `max` effort without a separate protocol change.

## Existing Contract

The Rust implementation follows the policy already encoded by pull request 85:

| Request field | Enabled-thinking policy | Disabled-thinking policy |
| --- | --- | --- |
| `thinking.type` | Must be in `--kimi-allowed-thinking-types`; omitted defaults to `enabled` | Must be explicitly allowed |
| `temperature` | Must be `1.0`; omitted defaults to `1.0` | Must be `0.6`; omitted defaults to `0.6` |
| `top_p` | Must be in `--kimi-allowed-top-p`; omitted prefers `1.0` when allowed, otherwise the first configured value | Same |
| `reasoning_effort` or `thinking.effort` | Must be in `--kimi-allowed-reasoning-efforts`; omitted defaults to `--kimi-default-reasoning-effort` | Top-level `reasoning_effort` is rejected; nested effort is ignored |
| `thinking.keep` | `all` or omitted preserves all reasoning; `interleaved` does not | Ignored |
| `presence_penalty` | Must be `0.0`; omitted defaults to `0.0` | Same |
| `frequency_penalty` | Must be `0.0`; omitted defaults to `0.0` | Same |
| `n` | Must be `1`; omitted defaults to `1` | Same |
| completion limit | When both token-limit aliases are omitted, default to `--kimi-default-max-completion-tokens` | Same |

The existing flag defaults remain unchanged:

- `--kimi-default-max-completion-tokens=32768`
- `--kimi-allowed-thinking-types=enabled,disabled`
- `--kimi-default-reasoning-effort=max`
- `--kimi-allowed-reasoning-efforts=low,high,max`
- `--kimi-allowed-top-p=0.95,1.0`

A Kimi-only deployment can narrow these values. For example,
`--kimi-allowed-thinking-types=enabled`, `--kimi-allowed-reasoning-efforts=max`, and
`--kimi-allowed-top-p=0.95` enforce thinking on, effort `max`, and `top_p=0.95`.

Explicit values are distinguished from omitted values. In particular, an explicitly allowed
`top_p=0.0` must remain `0.0`; it must not be replaced through truthiness-based defaulting.

## Architecture

### Typed Configuration

Add `KimiApiComplianceConfig` to `FrontendApiConfig`, next to the existing Anthropic and streaming
dispatch groups. It holds the enabled flag, completion-token default, thinking-type allowlist,
reasoning-effort default and allowlist, and `top_p` allowlist.

`components/src/dynamo/frontend/main.py` passes the six existing `--kimi-*` values to
`EntrypointArgs`. The Python binding builds the typed Rust config, and `HttpServiceConfig` retains it
in shared service state. Construction validates non-empty allowlists, numeric ranges, known enum
values, and that the configured effort default belongs to its allowlist. Invalid deployment config
fails at startup.

This follows pull request 85's explicit configuration approach. No request path inspects a model
name to decide whether compliance applies.

### Shared Request Policy

Add a small Rust Kimi compliance module under the OpenAI HTTP service. It validates and defaults
Chat Completions requests only when the typed config is enabled.

The handlers apply the policy immediately after JSON deserialization and before request-template
sampling defaults. This order makes the Kimi defaults authoritative and ensures invalid values are
returned as HTTP 400 errors before processor selection or worker dispatch.

For Chat Completions, the policy updates `NvCreateChatCompletionRequest`. The existing
`normalize_reasoning_template_args` function then maps the normalized thinking fields to the native
K3 template arguments. The policy validates every explicit effort alias and injects the configured
default only when neither alias is present. The existing and required default is `max`. Existing
Rust normalization continues to own alias precedence and template argument construction.

### Python Processor Compatibility

Keep pull request 85's Python policy for direct vLLM-processor use. Requests entering through the
HTTP service are already normalized by Rust, so the Python pass becomes an idempotent validation
pass. Update Python omission checks to use `is None` semantics where necessary, including `top_p`,
so an explicit allowed zero is not replaced by a default.

The Rust processor path does not invoke any vLLM preprocessing or postprocessing as a result of this
change.

## Request Flow

1. Parse the OpenAI Chat Completions request.
2. Read `KimiApiComplianceConfig` from shared frontend state.
3. If disabled, continue without mutation.
4. If enabled, validate explicit values and apply omission defaults.
5. Apply the normal request template for fields still omitted.
6. Run existing thinking-to-template-argument normalization and generic validation.
7. Dispatch through the selected Rust or vLLM chat processor.
8. Preserve the existing tool-call and reasoning response parsers.

## Error Handling

Request-policy errors use HTTP 400 with the existing OpenAI `invalid_request_error` envelope. Error
messages identify the rejected field and the accepted value or allowlist. Invalid frontend flag
combinations fail during startup rather than remaining silently inert.

Worker failures and parser failures retain their current status and error mapping. Kimi compliance
validation does not run at the worker.

## Testing

Add focused coverage at three boundaries:

- Rust config tests verify all six Python values reach `FrontendApiConfig`, invalid configuration
  fails at startup, and disabled compliance is a no-op.
- Rust policy tests cover enabled and disabled thinking, effort default `max`, narrowed allowlists,
  omission defaults, explicit values, explicit allowed `top_p=0.0`, token-limit aliases, and HTTP
  400 errors.
- HTTP tests exercise Chat Completions through the default Rust processor. The live compliance
  matrix must reject disabled thinking and disallowed `top_p` values when configured to allow only
  enabled thinking and `top_p=0.95`.

Retain the pull request 85 Python tests and add the explicit-zero regression. Existing K3 tool-call,
reasoning, and structural-tag tests must remain unchanged and pass.

## Scope and Reviewability

The implementation changes only configuration plumbing, the OpenAI ingress policy, and focused
tests. It does not modify the K3 renderer or parser implementations. The pull request description
will state the `Summary` and `Validation`, use a Conventional Commit title, and all commits will be
signed with DCO.
