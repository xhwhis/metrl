SELECT count(*) > 0 FROM pg_stat_activity
WHERE datname = current_database() AND wait_event = 'PgSleep'
