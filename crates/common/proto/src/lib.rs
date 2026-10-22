//! Internal gRPC protocol between servers and workers, generated from `proto/`.

#[allow(clippy::doc_markdown)]
pub mod worker {
    tonic::include_proto!("metrl.worker");
}

#[allow(clippy::doc_markdown)]
pub mod peer {
    tonic::include_proto!("metrl.peer");
}
