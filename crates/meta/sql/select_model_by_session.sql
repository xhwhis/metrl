SELECT
    id, org_id, session_id, session_seq, base, base_revision, arch, tokenizer_id, lora_rank, seed
FROM models
WHERE org_id = $1 AND session_id = $2 AND session_seq = $3
