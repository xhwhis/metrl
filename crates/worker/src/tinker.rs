//! Tinker wire format of forward-backward inputs and outputs.

use std::collections::HashMap;

use bytes::Bytes;
use error_stack::Report;

use crate::tinker_public::{self, chunk, tensor};
use crate::{DType, Datum, ForwardBackward, ForwardBackwardOutput, Tensor, WorkerError};

/// The SDK names the output class by this tag.
const OUTPUT_TYPE: &str = "ArrayRecord";

/// Decodes the Tinker `ForwardBackwardRequest` a server stored as a task's
/// input. Rejects inputs this worker cannot run as given rather than ignoring
/// part of them.
pub(crate) fn decode(input: &[u8]) -> Result<ForwardBackward, Report<WorkerError>> {
    use error_stack::ResultExt;
    use prost::Message;
    let request =
        tinker_public::ForwardBackwardRequest::decode(input).change_context_lazy(|| {
            WorkerError::Invalid("forward-backward input is not a Tinker request".to_owned())
        })?;
    if !request.loss_fn_config.is_empty() || !request.loss_fn_config_v2.is_empty() {
        return Err(Report::new(WorkerError::Invalid(
            "loss_fn_config is not supported".to_owned(),
        )));
    }
    let data = request
        .data
        .into_iter()
        .enumerate()
        .map(|(index, datum)| self::datum(index, datum))
        .collect::<Result<_, _>>()?;
    Ok(ForwardBackward {
        data,
        loss_fn: request.loss_fn,
        forward_only: request.forward_only,
    })
}

fn datum(index: usize, datum: tinker_public::Datum) -> Result<Datum, Report<WorkerError>> {
    let chunks: Vec<Vec<i32>> = datum
        .model_input
        .into_iter()
        .map(|chunk| {
            match chunk.chunk {
                Some(chunk::Chunk::EncodedText(text)) => {
                    let (tokens, []) = text.tokens.as_chunks::<4>() else {
                        return Err(Report::new(WorkerError::Invalid(format!(
                            "data[{index}].model_input: tokens are not int32"
                        ))));
                    };
                    Ok(tokens.iter().copied().map(i32::from_le_bytes).collect())
                }
                Some(chunk::Chunk::Image(_) | chunk::Chunk::Dmel(_)) | None => {
                    Err(Report::new(WorkerError::Invalid(format!(
                        "data[{index}].model_input: only encoded text is supported"
                    ))))
                }
            }
        })
        .collect::<Result<_, _>>()?;
    let loss_fn_inputs = datum
        .loss_fn_inputs
        .into_iter()
        .map(|(name, tensor)| {
            self::tensor(tensor)
                .map(|tensor| (name.clone(), tensor))
                .map_err(|report| {
                    let message = format!(
                        "data[{index}].loss_fn_inputs[{name}]: {}",
                        report.current_context()
                    );
                    report.change_context(WorkerError::Invalid(message))
                })
        })
        .collect::<Result<_, _>>()?;
    Ok(Datum {
        tokens: chunks.concat(),
        loss_fn_inputs,
    })
}

fn tensor(tensor: tinker_public::Tensor) -> Result<Tensor, Report<WorkerError>> {
    use error_stack::ResultExt;
    let dtype = match tensor.dtype() {
        tinker_public::DType::DtypeFloat32 => DType::Float32,
        tinker_public::DType::DtypeInt64 => DType::Int64,
        other => {
            return Err(Report::new(WorkerError::Invalid(format!(
                "dtype {other:?} is not supported"
            ))));
        }
    };
    let Some(tensor::Encoding::Dense(data)) = tensor.encoding else {
        return Err(Report::new(WorkerError::Invalid(
            "only dense tensors are supported".to_owned(),
        )));
    };
    // The SDK leaves the shape out of one-dimensional tensors.
    let shape = if tensor.shape.is_empty() {
        vec![data.len() / dtype.size()]
    } else {
        tensor
            .shape
            .iter()
            .map(|dim| usize::try_from(*dim))
            .collect::<Result<_, _>>()
            .change_context_lazy(|| WorkerError::Invalid("shape is negative".to_owned()))?
    };
    Tensor::new(Bytes::from(data), dtype, shape)
}

/// Encodes `output` as a Tinker `ForwardBackwardOutput` with one record that
/// batches every datum. Fails when the outputs cannot batch: every datum
/// must have the same fields, each with the same dtype and trailing shape.
pub(crate) fn encode(output: ForwardBackwardOutput) -> Result<Bytes, Report<WorkerError>> {
    use prost::Message;
    let invalid = |reason: String| {
        Report::new(WorkerError::Failed).attach(format!("engine output cannot batch: {reason}"))
    };
    if let Some(key) = output
        .metrics
        .keys()
        .find(|key| key.split(':').count() != 2)
    {
        return Err(invalid(format!("metric {key} is not name:reduction")));
    }
    let names: Vec<String> = output
        .loss_fn_outputs
        .first()
        .map_or_else(Vec::new, |first| first.keys().cloned().collect());
    if let Some(index) = output
        .loss_fn_outputs
        .iter()
        .position(|fields| !fields.keys().eq(names.iter()))
    {
        return Err(invalid(format!(
            "datum {index} has other fields than datum 0"
        )));
    }
    let fields = names
        .iter()
        .map(|name| {
            let tensors: Vec<&Tensor> = output
                .loss_fn_outputs
                .iter()
                .map(|fields| &fields[name])
                .collect();
            let first = tensors[0];
            // A scalar contributes one value per datum. The batched wire
            // format represents that as an empty trailing shape, just like
            // a one-dimensional tensor does when its leading dimension is
            // the number of values in that datum.
            let trailing = first.shape().get(1..).unwrap_or_default();
            if let Some(index) = tensors.iter().position(|tensor| {
                tensor.dtype() != first.dtype()
                    || tensor.shape().get(1..).unwrap_or_default() != trailing
            }) {
                return Err(invalid(format!(
                    "{name} of datum {index} differs in dtype or trailing shape"
                )));
            }
            let offsets = tensors.iter().scan(0_i64, |offset, tensor| {
                *offset += tensor.data().len() as i64;
                Some(*offset)
            });
            let batched = tinker_public::BatchedTensor {
                data: tensors
                    .iter()
                    .flat_map(|tensor| tensor.data().iter())
                    .copied()
                    .collect(),
                offsets: std::iter::once(0)
                    .chain(offsets)
                    .flat_map(i64::to_le_bytes)
                    .collect(),
                dtype: match first.dtype() {
                    DType::Float32 => tinker_public::DType::DtypeFloat32,
                    DType::Int64 => tinker_public::DType::DtypeInt64,
                } as i32,
                trailing_shape: trailing.iter().map(|dim| *dim as i64).collect(),
            };
            Ok((name.clone(), batched))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;
    let record = tinker_public::ArrayRecord {
        type_tag: OUTPUT_TYPE.to_owned(),
        fields,
        num_datums: output.loss_fn_outputs.len() as i64,
    };
    let encoded = tinker_public::ForwardBackwardOutput {
        loss_fn_output_type: OUTPUT_TYPE.to_owned(),
        loss_fn_outputs: vec![record],
        metrics: output.metrics.into_iter().collect(),
    };
    Ok(Bytes::from(encoded.encode_to_vec()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use prost::Message;

    use super::*;

    #[test]
    fn encodes_scalar_outputs_as_batched_values() {
        let scalar = |value: f32| {
            Tensor::new(
                Bytes::from(value.to_le_bytes().to_vec()),
                DType::Float32,
                Vec::new(),
            )
            .expect("one float fills a scalar tensor")
        };
        let output = ForwardBackwardOutput {
            loss_fn_outputs: vec![
                BTreeMap::from([(String::from("loss"), scalar(1.0))]),
                BTreeMap::from([(String::from("loss"), scalar(2.0))]),
            ],
            metrics: BTreeMap::new(),
        };

        let encoded = encode(output).expect("scalar fields are valid batched outputs");
        let decoded = tinker_public::ForwardBackwardOutput::decode(encoded)
            .expect("encoded output is valid Tinker protobuf");
        let field = &decoded.loss_fn_outputs[0].fields["loss"];
        assert!(
            field.trailing_shape.is_empty(),
            "scalar outputs have no trailing dimensions"
        );
        assert_eq!(decoded.loss_fn_outputs[0].num_datums, 2);
        assert_eq!(field.data, [0, 0, 128, 63, 0, 0, 0, 64]);
        assert_eq!(
            field.offsets,
            [
                0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0
            ]
        );
    }
}
