-- True once an outcome commit waits on a row lock.
SELECT count(*) > 0 FROM pg_stat_activity
WHERE datname = current_database() AND wait_event_type = 'Lock'
    AND query LIKE '%UPDATE tasks SET outcome%'
