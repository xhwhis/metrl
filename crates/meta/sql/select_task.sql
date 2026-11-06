SELECT model_id, seq, request_id, org_id, input, outcome, created_at FROM tasks
WHERE model_id = $1 AND seq = $2 AND org_id = $3
