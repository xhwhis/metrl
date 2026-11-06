-- Records a task as failed before it ran, only for the owning generation and
-- only while it is pending. `owned` is false once a newer generation took the
-- model over.
WITH owner AS (
    SELECT FROM models WHERE id = $1 AND generation = $2 AND org_id = $5 FOR SHARE
),
recorded AS (
    UPDATE tasks SET outcome = $4
    WHERE model_id = $1 AND seq = $3 AND org_id = $5 AND outcome IS NULL
        AND EXISTS (SELECT FROM owner)
)
SELECT EXISTS (SELECT FROM owner) AS owned
