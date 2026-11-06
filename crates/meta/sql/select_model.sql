SELECT
    id, org_id, session_id, session_seq, base, base_revision, arch, tokenizer_id, lora_rank, seed
FROM models
WHERE id = $1 AND org_id = $2
