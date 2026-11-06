/// SHA-256 of an API key. Only the hash is stored or compared.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KeyHash(pub [u8; 32]);

impl KeyHash {
    pub fn of(key: &str) -> Self {
        use sha2::Digest;
        Self(sha2::Sha256::digest(key.as_bytes()).into())
    }
}
