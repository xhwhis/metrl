SELECT pg_terminate_backend(pid, 5000) FROM pg_stat_activity
WHERE backend_type = 'walsender' AND datname = current_database()
