# pydantic_to_flat_dict_and_back

A Python package that converts nested Pydantic models to flat dictionaries and back, making it easier to work with nested data structures.

## Installation

```bash
pip install pydantic_to_flat_dict_and_back
```

## Features

- Convert nested Pydantic models to flat dictionaries
- Convert flat dictionaries back to nested Pydantic models
- Optional prefix handling for nested fields
- Compatibility checking for models

## Requirements

- Python >=3.12
- Pydantic ~2.6.1

## Usage

### Basic Example

```python
import pydantic
from pydantic_to_flat_dict_and_back import pydantic_to_flat_dict, flat_dict_to_pydantic

class A(pydantic.BaseModel):
    value: float
    name: str
    seed: int
    model_config = pydantic.ConfigDict(extra="ignore")

class B(pydantic.BaseModel):
    value: float
    name: str
    a: A
    model_config = pydantic.ConfigDict(extra="ignore")

# Create a nested model
model = B(
    value=2.0, 
    name="nested", 
    a=A(value=3.0, name="nested2", seed=42)
)

# Convert to flat dictionary
flat_dict = pydantic_to_flat_dict(model, prefix_nested=True)
print(flat_dict)
# Output: {'value': 2.0, 'name': 'nested', 'a_value': 3.0, 'a_name': 'nested2', 'a_seed': 42}

# Convert back to nested model
reconstructed = flat_dict_to_pydantic(B, flat_dict, prefix_nested=True)
assert model.model_dump() == reconstructed.model_dump()
```

### Important Notes

1. All models must have `ConfigDict(extra="ignore")` set
2. Field aliases are not supported
3. When `prefix_nested=True`, nested fields are prefixed with their parent field name

## API Reference

### pydantic_to_flat_dict

```python
def pydantic_to_flat_dict(
    model: pydantic.BaseModel, 
    prefix_nested: bool = False
) -> dict[str, Any]
```

Converts a nested Pydantic model to a flat dictionary.

### flat_dict_to_pydantic

```python
def flat_dict_to_pydantic(
    model_cls: Type[PydanticType], 
    flat_dict: dict[str, Any], 
    prefix_nested: bool = False
) -> PydanticType
```

Reconstructs a Pydantic model from a flat dictionary.

### check_pydantic_model_compatibility

```python
def check_pydantic_model_compatibility(
    model_cls: Type[PydanticType]
) -> tuple[bool, list[str]]
```

Checks if a model is compatible with the flattening/unflattening operations.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.