SELECT pg_logical_emit_message(false, $1::text, $2::text, flush => true)
