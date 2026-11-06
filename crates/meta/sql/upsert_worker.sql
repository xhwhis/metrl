INSERT INTO workers (id, addr, bases) VALUES ($1, $2, $3)
ON CONFLICT (id) DO UPDATE
SET addr = excluded.addr, bases = excluded.bases
-- An unchanged periodic re-registration only locks the row, so it publishes no change.
WHERE (workers.addr, workers.bases) IS DISTINCT FROM (excluded.addr, excluded.bases)
