"""Regenerate proto/tinker.proto from the installed tinker SDK's wire descriptor."""

import sys
from pathlib import Path

from google.protobuf import descriptor_pb2
from tinker.proto import tinker_public_pb2

SCALARS = {
    descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE: "double",
    descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT: "float",
    descriptor_pb2.FieldDescriptorProto.TYPE_INT64: "int64",
    descriptor_pb2.FieldDescriptorProto.TYPE_UINT64: "uint64",
    descriptor_pb2.FieldDescriptorProto.TYPE_INT32: "int32",
    descriptor_pb2.FieldDescriptorProto.TYPE_BOOL: "bool",
    descriptor_pb2.FieldDescriptorProto.TYPE_STRING: "string",
    descriptor_pb2.FieldDescriptorProto.TYPE_BYTES: "bytes",
    descriptor_pb2.FieldDescriptorProto.TYPE_UINT32: "uint32",
    descriptor_pb2.FieldDescriptorProto.TYPE_SINT32: "sint32",
    descriptor_pb2.FieldDescriptorProto.TYPE_SINT64: "sint64",
}


def type_name(field: descriptor_pb2.FieldDescriptorProto, package: str) -> str:
    if field.type_name:
        return field.type_name.removeprefix(f".{package}.")
    return SCALARS[field.type]


def emit_enum(enum: descriptor_pb2.EnumDescriptorProto, indent: str, out: list[str]) -> None:
    out.append(f"{indent}enum {enum.name} {{")
    out.extend(f"{indent}  {value.name} = {value.number};" for value in enum.value)
    out.append(f"{indent}}}")


def emit_message(
    message: descriptor_pb2.DescriptorProto, package: str, indent: str, out: list[str]
) -> None:
    out.append(f"{indent}message {message.name} {{")
    map_entries = {
        nested.name: nested for nested in message.nested_type if nested.options.map_entry
    }
    for nested in message.nested_type:
        if not nested.options.map_entry:
            emit_message(nested, package, indent + "  ", out)
    for enum in message.enum_type:
        emit_enum(enum, indent + "  ", out)
    emitted_oneofs: set[int] = set()
    for field in message.field:
        in_oneof = field.HasField("oneof_index") and not field.proto3_optional
        if in_oneof:
            if field.oneof_index in emitted_oneofs:
                continue
            emitted_oneofs.add(field.oneof_index)
            out.append(f"{indent}  oneof {message.oneof_decl[field.oneof_index].name} {{")
            out.extend(
                f"{indent}    {type_name(member, package)} {member.name} = {member.number};"
                for member in message.field
                if member.HasField("oneof_index") and member.oneof_index == field.oneof_index
            )
            out.append(f"{indent}  }}")
            continue
        entry = map_entries.get(field.type_name.rsplit(".", 1)[-1])
        if entry is not None:
            key, value = entry.field
            out.append(
                f"{indent}  map<{type_name(key, package)}, {type_name(value, package)}> "
                f"{field.name} = {field.number};"
            )
            continue
        if field.label == descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED:
            label = "repeated "
        elif field.proto3_optional:
            label = "optional "
        else:
            label = ""
        out.append(f"{indent}  {label}{type_name(field, package)} {field.name} = {field.number};")
    out.append(f"{indent}}}")


def main() -> None:
    file = descriptor_pb2.FileDescriptorProto()
    tinker_public_pb2.DESCRIPTOR.CopyToProto(file)
    out = ['syntax = "proto3";', "", f"package {file.package};", ""]
    for enum in file.enum_type:
        emit_enum(enum, "", out)
        out.append("")
    for message in file.message_type:
        emit_message(message, file.package, "", out)
        out.append("")
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("proto/tinker.proto")
    target.write_text("\n".join(out).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
