# scheduler

Pure decision logic. Given the same state and events, it must produce the same effects;
recovery depends on this.

- No I/O, no async, no clock reads. Time arrives as `now`; timers are expressed through
  `next_deadline(&State)` and come back as `Event::Tick`.
- Depends only on `common/*`. Never on `meta`, `object`, `server`, or tokio I/O.
- `model/`: per-model task log, seq ordering and dedupe, gap timeout, replay.
  `cluster/`: worker placement, adapter affinity, load, per-org fairness.
- Seq rules: execute in `seq` order and never skip one. A failed task is a no-op that
  advances `applied`; replay skips it. Replay never fires timeouts.
- New behavior is a new `Event` or `Effect` variant, not a side channel.
- Unit tests are welcome here for ordering, gaps, replay, and fairness, which are hard
  to reach end to end. Test through `on_event`, not internal functions.
