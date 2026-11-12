//! A saved adapter in PEFT layout, as samplers load it. A checkpoint is
//! complete only once its `manifest.json` exists, so that file is written last.

use std::collections::BTreeMap;

use bytes::{BufMut, Bytes, BytesMut};
use error_stack::Report;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{AdapterWeights, DType, Tensor, WorkerError};

const MODEL: &str = "adapter_model.safetensors";
const CONFIG: &str = "adapter_config.json";
const MANIFEST: &str = "manifest.json";
const METADATA: &str = "__metadata__";

/// One file of a saved adapter, named relative to its checkpoint's prefix.
pub(crate) struct File {
    pub(crate) name: &'static str,
    pub(crate) body: Bytes,
}

/// The adapter's files in the order they must be written.
pub(crate) fn files(
    weights: &AdapterWeights, base: &str,
) -> Result<Vec<File>, Report<WorkerError>> {
    // JSON has no NaN or infinity.
    if !weights.alpha.is_finite() {
        return Err(Report::new(WorkerError::Failed)
            .attach(format!("engine returned lora alpha {}", weights.alpha)));
    }
    let config = json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base,
        "r": weights.rank,
        "lora_alpha": weights.alpha,
        "target_modules": weights.target_modules,
        "bias": "none",
    });
    let manifest = json!({
        "base": base,
        "lora_rank": weights.rank,
        "alpha": weights.alpha,
        "files": [MODEL, CONFIG],
    });
    Ok(vec![
        File {
            name: MODEL,
            body: safetensors(weights),
        },
        File {
            name: CONFIG,
            body: Bytes::from(config.to_string()),
        },
        File {
            name: MANIFEST,
            body: Bytes::from(manifest.to_string()),
        },
    ])
}

/// The header's length as a little-endian u64, the JSON header, then each
/// tensor's bytes in name order, at offsets relative to the end of the header.
fn safetensors(weights: &AdapterWeights) -> Bytes {
    let mut header = Map::new();
    header.insert(METADATA.to_owned(), json!({"format": "pt"}));
    let mut end = 0;
    for (name, tensor) in &weights.tensors {
        let start = end;
        end += tensor.data().len();
        header.insert(
            name.clone(),
            json!({
                "dtype": dtype(tensor.dtype()),
                "shape": tensor.shape(),
                "data_offsets": [start, end],
            }),
        );
    }
    let mut header = Value::Object(header).to_string();
    // Readers may map the file, so the tensor data starts 8-byte aligned.
    let padding = header.len().next_multiple_of(8) - header.len();
    header.extend(std::iter::repeat_n(' ', padding));
    let mut file = BytesMut::with_capacity(8 + header.len() + end);
    file.put_u64_le(header.len() as u64);
    file.put_slice(header.as_bytes());
    for tensor in weights.tensors.values() {
        file.put_slice(tensor.data());
    }
    file.freeze()
}

fn dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 => "F32",
        DType::Int64 => "I64",
    }
}

#[derive(Deserialize)]
struct Config {
    r: u32,
    lora_alpha: f64,
    target_modules: Vec<String>,
}

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// The adapter saved under `prefix`. Its checkpoint row is committed only
/// after every file was written, so a missing or unreadable file is a failure,
/// not a save still in progress.
pub(crate) async fn load(
    object: &metrl_object::Store, prefix: &str,
) -> Result<AdapterWeights, Report<WorkerError>> {
    use error_stack::ResultExt;
    let get = |name: &str| {
        let key = format!("{prefix}{name}");
        async move { object.get(&key).await.change_context(WorkerError::Failed) }
    };
    let (model, config) = tokio::try_join!(get(MODEL), get(CONFIG))?;
    let config: Config = serde_json::from_slice(&config)
        .change_context(WorkerError::Failed)
        .attach_with(|| format!("parse {prefix}{CONFIG}"))?;
    let tensors = decode(&model).attach_with(|| format!("decode {prefix}{MODEL}"))?;
    Ok(AdapterWeights {
        rank: config.r,
        alpha: config.lora_alpha,
        target_modules: config.target_modules,
        tensors,
    })
}

/// Each tensor's data is a slice of `file`, not a copy.
fn decode(file: &Bytes) -> Result<BTreeMap<String, Tensor>, Report<WorkerError>> {
    use error_stack::ResultExt;
    let corrupt = |what: &str| Report::new(WorkerError::Failed).attach(what.to_owned());
    let length = file
        .first_chunk::<8>()
        .map(|length| u64::from_le_bytes(*length))
        .ok_or_else(|| corrupt("no header length"))?;
    let start = usize::try_from(length)
        .ok()
        .and_then(|length| length.checked_add(8))
        .filter(|start| *start <= file.len())
        .ok_or_else(|| corrupt("header runs past the end of the file"))?;
    let header: Map<String, Value> =
        serde_json::from_slice(&file[8..start]).change_context(WorkerError::Failed)?;
    let data = file.slice(start..);
    header
        .into_iter()
        .filter(|(name, _)| name != METADATA)
        .map(|(name, entry)| {
            let entry: Entry = serde_json::from_value(entry)
                .change_context(WorkerError::Failed)
                .attach_with(|| format!("tensor {name}"))?;
            let dtype = match entry.dtype.as_str() {
                "F32" => DType::Float32,
                "I64" => DType::Int64,
                other => return Err(corrupt(&format!("tensor {name} has dtype {other}"))),
            };
            let [begin, end] = entry.data_offsets;
            if begin > end || end > data.len() {
                return Err(corrupt(&format!("tensor {name} runs past the data")));
            }
            let tensor = Tensor::new(data.slice(begin..end), dtype, entry.shape)
                .change_context(WorkerError::Failed)
                .attach_with(|| format!("tensor {name}"))?;
            Ok((name, tensor))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Samplers load the file with a safetensors reader; a wrong offset or an
    /// unaligned data section would load the wrong weights or none.
    #[test]
    fn safetensors_reader_finds_every_tensor() {
        let tensor = |byte: u8, dtype, shape: Vec<usize>| {
            let len = shape.iter().product::<usize>() * 8;
            let data = Bytes::from(vec![byte; len]);
            Tensor::new(data, dtype, shape).expect("eight bytes per element fill the shape")
        };
        let weights = AdapterWeights {
            rank: 2,
            alpha: 4.0,
            target_modules: vec!["q_proj".to_owned()],
            tensors: BTreeMap::from([
                (
                    "b.lora_B.weight".to_owned(),
                    tensor(1, DType::Int64, vec![3, 2]),
                ),
                (
                    "a.lora_A.weight".to_owned(),
                    tensor(2, DType::Int64, vec![2, 1]),
                ),
            ]),
        };
        let file = safetensors(&weights);
        let (length, rest) = file.split_at(8);
        let length = u64::from_le_bytes(length.try_into().expect("the prefix is 8 bytes"));
        let length = usize::try_from(length).expect("the header fits in memory");
        assert_eq!(length % 8, 0, "the tensor data must start 8-byte aligned");
        let (header, data) = rest.split_at(length);
        let header: Value = serde_json::from_slice(header).expect("the header is JSON");
        let mut total = 0;
        for (name, tensor) in &weights.tensors {
            let entry = &header[name.as_str()];
            let offset = |index: usize| {
                let offset = entry["data_offsets"][index]
                    .as_u64()
                    .expect("offsets are integers");
                usize::try_from(offset).expect("offsets fit in memory")
            };
            assert_eq!(
                data.get(offset(0)..offset(1)),
                Some(tensor.data().as_ref()),
                "{name} must point at its own bytes"
            );
            assert_eq!(entry["dtype"], "I64", "{name} must keep its dtype");
            assert_eq!(
                entry["shape"],
                json!(tensor.shape()),
                "{name} must keep its shape"
            );
            total += tensor.data().len();
        }
        assert_eq!(
            total,
            data.len(),
            "the data must hold the tensors and nothing else"
        );
        let decoded = decode(&file).expect("the file was just encoded");
        assert_eq!(
            decoded.keys().collect::<Vec<_>>(),
            weights.tensors.keys().collect::<Vec<_>>(),
            "decoding must find every tensor and no metadata"
        );
        for (name, tensor) in &weights.tensors {
            let loaded = &decoded[name];
            assert_eq!(
                (loaded.data(), loaded.dtype(), loaded.shape()),
                (tensor.data(), tensor.dtype(), tensor.shape()),
                "{name} must load as it was saved"
            );
        }
    }
}
