"""A small JSON-Schema subset validator (types, required, properties, enum,
items, min/max). Shared by tool argument validation (FR-404) and the
JSON-schema eval evaluator (FR-903)."""

_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def validate(instance, schema, path: str = "$") -> list[str]:
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors
    t = schema.get("type")
    if t:
        py = _TYPE_MAP.get(t)
        if py is not None:
            # bool is a subclass of int in Python — keep them distinct.
            if t in ("number", "integer") and isinstance(instance, bool):
                errors.append(f"{path}: expected {t}, got boolean")
            elif not isinstance(instance, py):
                errors.append(f"{path}: expected {t}, got {type(instance).__name__}")
                return errors
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in enum {schema['enum']}")
    if isinstance(instance, dict):
        for req in schema.get("required", []):
            if req not in instance:
                errors.append(f"{path}.{req}: required property missing")
        props = schema.get("properties", {})
        for key, subschema in props.items():
            if key in instance:
                errors.extend(validate(instance[key], subschema, f"{path}.{key}"))
    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
    return errors
