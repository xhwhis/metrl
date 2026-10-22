# 0008. Checkpoints

## Context

Saves are the durable snapshots users rely on, and auto checkpoints bound recovery.
Object stores have no transactions, so a crash mid-write must never look like a valid
checkpoint.

## Decision

- Objects are write-once under `orgs/{org_id}/models/{model}/...`.
- Payload files are written first and `manifest.json` last.
- A checkpoint exists only when its manifest exists and a Postgres row points to it.
- `tinker://` paths are logical and resolve through the `checkpoints` table.

## Alternatives

- Overwrite in place: readers can see partial state.
- Listing the object store as the index: not authoritative and slow.

## Consequences

Orphaned objects from failed writes are harmless and cleaned by sweep. Renaming or
moving a checkpoint is a row update, not an object copy.
