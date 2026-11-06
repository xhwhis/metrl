-- Records a task's outcome and advances `applied`, both only for the owning
-- generation. `owned` is false once a newer generation took the model over. A
-- task that is already applied, or whose outcome is already recorded, records
-- nothing; the log keeps the outcome it has.
WITH owner AS (
    SELECT applied FROM models
    WHERE id = $1 AND generation = $2 AND org_id = $5
    FOR NO KEY UPDATE
),
recorded AS (
    UPDATE tasks SET outcome = $4
    WHERE model_id = $1 AND seq = $3 AND org_id = $5 AND outcome IS NULL
        AND EXISTS (SELECT FROM owner WHERE applied < $3)
    RETURNING seq
),
advanced AS (
    UPDATE models SET applied = $3
    WHERE id = $1 AND generation = $2 AND org_id = $5 AND applied < $3
        AND EXISTS (SELECT FROM recorded WHERE seq = $3)
)
SELECT EXISTS (SELECT FROM owner) AS owned
