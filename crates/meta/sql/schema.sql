CREATE TABLE orgs (
    -- Object keys start with `orgs/{id}/`, so an id must not reach outside its
    -- prefix.
    id text PRIMARY KEY CHECK (id ~ '^[A-Za-z0-9_-]+$'),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE api_keys (
    -- sha256 of the key.
    key_hash bytea PRIMARY KEY CHECK (octet_length(key_hash) = 32),
    org_id text NOT NULL REFERENCES orgs (id),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE workers (
    id uuid PRIMARY KEY,
    addr text NOT NULL,
    bases jsonb NOT NULL,
    registered_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE models (
    -- Also the request id of the `create_model` call that created it.
    id uuid PRIMARY KEY,
    org_id text NOT NULL REFERENCES orgs (id),
    UNIQUE (id, org_id),
    -- A retried `create_model` finds the model it created first.
    session_id uuid NOT NULL,
    session_seq bigint NOT NULL,
    UNIQUE (org_id, session_id, session_seq),
    base text NOT NULL,
    base_revision text NOT NULL,
    arch text NOT NULL,
    tokenizer_id text NOT NULL,
    lora_rank integer NOT NULL CHECK (lora_rank > 0),
    seed bigint NOT NULL,
    -- Lease and fencing state, renewed every few seconds, so `models` is not
    -- published.
    owner_server uuid,
    lease_until timestamptz,
    generation bigint NOT NULL DEFAULT 0,
    -- Every task up to this seq has run or failed.
    applied bigint NOT NULL DEFAULT 0,
    -- The trainer that holds the model's state.
    worker_id uuid,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE tasks (
    model_id uuid NOT NULL,
    seq bigint NOT NULL CHECK (seq > 0),
    request_id uuid NOT NULL UNIQUE,
    org_id text NOT NULL REFERENCES orgs (id),
    input jsonb NOT NULL,
    -- NULL while pending.
    outcome jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, seq),
    FOREIGN KEY (model_id, org_id) REFERENCES models (id, org_id)
);

CREATE PUBLICATION metrl FOR TABLE api_keys, workers, tasks;
