-- A worker restarted on the same address replaces its previous registration.
DELETE FROM workers WHERE addr = $1 AND id <> $2
