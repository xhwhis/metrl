fn main() {
    tonic_prost_build::configure()
        // The default keys off `CARGO`, which maturin unsets; without it, proto edits never
        // rebuild.
        .emit_rerun_if_changed(true)
        .build_server(false)
        .build_client(false)
        .compile_protos(&["../../proto/tinker.proto"], &["../../proto"])
        .expect("tinker.proto compiles with the system protoc");
}
