fn main() {
    println!("cargo:rerun-if-changed=../../proto");
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(false)
        .compile_protos(&["../../proto/tinker.proto"], &["../../proto"])
        .expect("tinker.proto compiles with the system protoc");
}
