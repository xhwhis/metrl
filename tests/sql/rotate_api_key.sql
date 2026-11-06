UPDATE api_keys SET key_hash = sha256(convert_to(:'new', 'UTF8'))
WHERE key_hash = sha256(convert_to(:'old', 'UTF8'))
