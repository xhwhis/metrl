//! Decoder for the `pgoutput` logical replication protocol, version 1, with
//! binary column values.

use bytes::{Buf, Bytes};
use error_stack::{Report, ResultExt};

use crate::MetaError;

pub(crate) struct Relation {
    pub(crate) id: u32,
    pub(crate) table: String,
    pub(crate) columns: Vec<Column>,
}

pub(crate) struct Column {
    pub(crate) name: String,
    /// Resolved only when the column is read, so a column of a type Metrl
    /// does not know fails nothing until something reads it.
    pub(crate) type_oid: u32,
}

pub(crate) enum Value {
    Null,
    /// An out-of-line (TOAST) value the change did not modify, so it was not
    /// sent.
    Unchanged,
    Binary(Bytes),
}

pub(crate) enum Message {
    Relation(Relation),
    Insert {
        relation: u32,
        row: Vec<Value>,
    },
    Update {
        relation: u32,
        /// The old key, sent only when the update changed it or the table has
        /// `REPLICA IDENTITY FULL`.
        old_key: Option<Vec<Value>>,
        row: Vec<Value>,
    },
    Delete {
        relation: u32,
        key: Vec<Value>,
    },
    Truncate {
        relations: Vec<u32>,
    },
    /// Messages Metrl does not act on: types, origins, and logical messages.
    Ignored,
}

pub(crate) fn decode(mut data: Bytes) -> Result<Message, Report<MetaError>> {
    let tag = data.try_get_u8().change_context(MetaError::Cdc)?;
    let message = match tag {
        b'R' => {
            let id = data.try_get_u32().change_context(MetaError::Cdc)?;
            let _namespace = string(&mut data)?;
            let table = string(&mut data)?;
            let _replica_identity = data.try_get_u8().change_context(MetaError::Cdc)?;
            let count = data.try_get_i16().change_context(MetaError::Cdc)?;
            let columns = (0..count)
                .map(|_| {
                    let _flags = data.try_get_u8().change_context(MetaError::Cdc)?;
                    let name = string(&mut data)?;
                    let type_oid = data.try_get_u32().change_context(MetaError::Cdc)?;
                    let _modifier = data.try_get_i32().change_context(MetaError::Cdc)?;
                    Ok(Column { name, type_oid })
                })
                .collect::<Result<_, Report<MetaError>>>()?;
            Message::Relation(Relation { id, table, columns })
        }
        b'I' => {
            let relation = data.try_get_u32().change_context(MetaError::Cdc)?;
            if data.try_get_u8().change_context(MetaError::Cdc)? != b'N' {
                return Err(Report::new(MetaError::Cdc).attach("insert without a new tuple"));
            }
            Message::Insert {
                relation,
                row: tuple(&mut data)?,
            }
        }
        b'U' => {
            let relation = data.try_get_u32().change_context(MetaError::Cdc)?;
            let mut marker = data.try_get_u8().change_context(MetaError::Cdc)?;
            let old_key = match marker {
                b'K' | b'O' => {
                    let key = tuple(&mut data)?;
                    marker = data.try_get_u8().change_context(MetaError::Cdc)?;
                    Some(key)
                }
                _ => None,
            };
            if marker != b'N' {
                return Err(Report::new(MetaError::Cdc).attach("update without a new tuple"));
            }
            Message::Update {
                relation,
                old_key,
                row: tuple(&mut data)?,
            }
        }
        b'D' => {
            let relation = data.try_get_u32().change_context(MetaError::Cdc)?;
            let marker = data.try_get_u8().change_context(MetaError::Cdc)?;
            if marker != b'K' && marker != b'O' {
                return Err(Report::new(MetaError::Cdc).attach("delete without a key tuple"));
            }
            Message::Delete {
                relation,
                key: tuple(&mut data)?,
            }
        }
        b'T' => {
            let count = data.try_get_u32().change_context(MetaError::Cdc)?;
            let _options = data.try_get_u8().change_context(MetaError::Cdc)?;
            let relations = (0..count)
                .map(|_| data.try_get_u32().change_context(MetaError::Cdc))
                .collect::<Result<_, _>>()?;
            Message::Truncate { relations }
        }
        _ => Message::Ignored,
    };
    Ok(message)
}

fn string(data: &mut Bytes) -> Result<String, Report<MetaError>> {
    let end = data
        .iter()
        .position(|&byte| byte == 0)
        .ok_or_else(|| Report::new(MetaError::Cdc).attach("unterminated string"))?;
    let value = data.split_to(end);
    data.advance(1);
    String::from_utf8(value.to_vec()).change_context(MetaError::Cdc)
}

fn tuple(data: &mut Bytes) -> Result<Vec<Value>, Report<MetaError>> {
    let count = data.try_get_i16().change_context(MetaError::Cdc)?;
    (0..count)
        .map(|_| {
            let kind = data.try_get_u8().change_context(MetaError::Cdc)?;
            match kind {
                b'n' => Ok(Value::Null),
                b'u' => Ok(Value::Unchanged),
                b'b' => {
                    let length = data.try_get_u32().change_context(MetaError::Cdc)? as usize;
                    if data.remaining() < length {
                        return Err(Report::new(MetaError::Cdc).attach("truncated column value"));
                    }
                    Ok(Value::Binary(data.split_to(length)))
                }
                _ => {
                    Err(Report::new(MetaError::Cdc)
                        .attach(format!("unknown column kind {:?}", char::from(kind))))
                }
            }
        })
        .collect()
}
