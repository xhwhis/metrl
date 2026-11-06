SELECT model_id, seq, request_id, org_id, input, outcome, created_at FROM tasks
WHERE request_id = $1 AND org_id = $2
