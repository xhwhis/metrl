fn main() {
    tonic_prost_build::configure()
        // The default keys off `CARGO`, which maturin unsets; without it, proto edits never
        // rebuild.
        .emit_rerun_if_changed(true)
        .compile_protos(
            &["../../../proto/worker.proto", "../../../proto/peer.proto"],
            &["../../../proto"],
        )
        .expect("internal protos compile with the system protoc");
}
