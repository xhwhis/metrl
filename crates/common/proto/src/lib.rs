//! Internal gRPC protocols between servers and workers, generated from
//! `proto/`.

#[allow(
    unused_qualifications,
    clippy::clone_on_ref_ptr,
    clippy::default_trait_access,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref
)]
pub mod worker {
    tonic::include_proto!("metrl.worker");
}

#[allow(
    unused_qualifications,
    clippy::clone_on_ref_ptr,
    clippy::default_trait_access,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref
)]
pub mod server {
    tonic::include_proto!("metrl.server");
}

#[allow(
    unused_qualifications,
    clippy::clone_on_ref_ptr,
    clippy::default_trait_access,
    clippy::doc_markdown,
    clippy::trivially_copy_pass_by_ref
)]
pub mod peer {
    tonic::include_proto!("metrl.peer");
}
