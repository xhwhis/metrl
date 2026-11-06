fn main() {
    println!("cargo:rerun-if-changed=../../../proto");
    tonic_prost_build::configure()
        .compile_protos(
            &[
                "../../../proto/worker.proto",
                "../../../proto/server.proto",
                "../../../proto/peer.proto",
            ],
            &["../../../proto"],
        )
        .expect("internal protos compile with the system protoc");
}
