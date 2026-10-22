# 0013. Base models

## Context

The SDK asks which base models exist and creates LoRA models on them. Only workers know
what they actually loaded.

## Decision

- Workers alone advertise bases, from `worker.toml` `[[bases]]` (`name`, `path`,
  `roles`, optional `max_context_length`).
- Registration reports `name`, `revision`, `arch`, `tokenizer_id`, `max_context_length`,
  `max_lora_rank`, and `roles`; servers store them in `workers.bases`.
- A different revision for an existing name is rejected.
- `get_server_capabilities` aggregates live workers. CreateModel returns 400 when no live
  worker serves the base. Models snapshot `base_model` and `base_revision`.

## Alternatives

- A server-side base registry: can advertise bases no worker serves.

## Consequences

Capabilities change as workers come and go. The external `name` is decoupled from the
loaded `path`.
