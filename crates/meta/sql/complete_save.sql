-- Records a `save_weights_for_sampler` outcome with the checkpoint it wrote,
-- and advances `applied`, all only for the owning generation. When another of
-- the model's checkpoints holds the name, records the name-taken outcome and
-- no checkpoint. `owned` is false once a newer generation took the model over.
-- A task that is already applied records nothing.
WITH owner AS (
    SELECT applied, base, base_revision, lora_rank FROM models
    WHERE id = $1 AND generation = $2 AND org_id = $6
    FOR NO KEY UPDATE
),
inserted AS (
    INSERT INTO checkpoints (
        id, org_id, model_id, seq, name, prefix, base, base_revision, lora_rank,
        size_bytes, ttl_seconds, expires_at, user_metadata
    )
    SELECT
        $7::uuid, $6::text, $1::uuid, $3::bigint, $8::text, $9::text,
        base, base_revision, lora_rank, $10::bigint, $11::bigint,
        now() + make_interval(secs => $11::bigint), $12::jsonb
    FROM owner
    WHERE applied < $3
    ON CONFLICT (model_id, name) DO NOTHING
    RETURNING id
),
recorded AS (
    UPDATE tasks
    SET outcome = CASE WHEN EXISTS (SELECT FROM inserted) THEN $4::jsonb ELSE $5::jsonb END
    WHERE model_id = $1 AND seq = $3 AND org_id = $6 AND outcome IS NULL
        AND EXISTS (SELECT FROM owner WHERE applied < $3)
    RETURNING seq
),
advanced AS (
    UPDATE models SET applied = $3
    WHERE id = $1 AND generation = $2 AND org_id = $6 AND applied < $3
        AND EXISTS (SELECT FROM recorded WHERE seq = $3)
)
SELECT EXISTS (SELECT FROM owner) AS owned
