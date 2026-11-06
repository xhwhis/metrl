-- Extends this server's leases and takes over every model whose lease is free
-- or expired. Taking a model over starts a new generation, which fences the
-- previous owner's writes.
UPDATE models
SET
    owner_server = $1,
    lease_until = now() + make_interval(secs => $2),
    generation = CASE
        WHEN owner_server = $1 AND lease_until > now() THEN generation
        ELSE generation + 1
    END
WHERE owner_server = $1 OR owner_server IS NULL OR lease_until <= now()
RETURNING id, org_id, base, base_revision, lora_rank, seed, generation, applied, worker_id
