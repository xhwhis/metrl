-- Holds the model's row lock until this backend is terminated, so its owner
-- can run a task on the trainer but cannot commit the outcome.
WITH locked AS MATERIALIZED (
    SELECT id FROM models WHERE id = :'model' FOR NO KEY UPDATE
)
SELECT pg_sleep(60) FROM locked
