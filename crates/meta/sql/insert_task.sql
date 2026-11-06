-- Appends a task to its model's log unless the seq is taken. The foreign key
-- fails when the model is not in the org.
INSERT INTO tasks (model_id, seq, request_id, org_id, input, created_at)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (model_id, seq) DO NOTHING
