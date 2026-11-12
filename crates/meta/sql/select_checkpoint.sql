SELECT id, org_id, model_id, name, prefix, base, base_revision FROM checkpoints
WHERE id = $1 AND org_id = $2
