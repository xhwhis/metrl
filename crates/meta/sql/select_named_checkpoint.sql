SELECT id, org_id, model_id, name, prefix, base, base_revision FROM checkpoints
WHERE model_id = $1 AND name = $2 AND org_id = $3
