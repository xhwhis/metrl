SELECT confirmed_flush_lsn >= :'lsn' FROM pg_replication_slots WHERE database = current_database()
