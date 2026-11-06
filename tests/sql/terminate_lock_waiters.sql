SELECT pg_terminate_backend(pid, 5000) FROM pg_stat_activity
WHERE datname = current_database() AND wait_event_type = 'Lock'
