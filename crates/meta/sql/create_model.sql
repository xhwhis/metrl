-- Creates a model unless the same session and model_seq_id already did.
INSERT INTO models (
    id, org_id, session_id, session_seq, base, base_revision, arch, tokenizer_id, lora_rank,
    seed
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (org_id, session_id, session_seq) DO NOTHING
