# %%
import inspect
from typing import Any, Type, TypeVar

import pydantic

PydanticType = TypeVar("PydanticType", bound=pydantic.BaseModel)


def pydantic_to_flat_dict(
    model: pydantic.BaseModel, prefix_nested: bool = False, _parent_dict: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Convert a Pydantic model with nested models into a flat dictionary.

    Args:
        model: The Pydantic model to flatten
        prefix_nested: If True, add prefixes to resolve field name collisions.
                        If False, raise an error when collisions occur.
        _parent_dict: Internal parameter for recursion tracking

    Returns:
        A flat dictionary with all nested fields

    Raises:
        ValueError: When prefix_nested is False and field name collisions are detected
    """
    if _parent_dict is None:
        _parent_dict = {}

    flat_dict = {}
    model_dict = model.model_dump()

    for field_name, field_value in model_dict.items():
        # Get the field type from the model's fields
        field_type = model.model_fields[field_name].annotation

        # If the field is another Pydantic model, recursively flatten it
        if hasattr(field_type, "model_fields"):
            nested_model = getattr(model, field_name)
            # Recursively flatten the nested model
            nested_flat = pydantic_to_flat_dict(nested_model, prefix_nested, flat_dict)

            # Add the nested fields to our flat dict
            for nested_key, nested_value in nested_flat.items():
                key = nested_key
                if key in flat_dict and not prefix_nested:
                    raise ValueError(
                        f"Field name collision detected: '{key}' exists in both "
                        f"{type(model).__name__} and a nested model. Set prefix_nested=True to resolve."
                    )

                if prefix_nested:
                    # Add prefix if there's a collision
                    key = f"{field_name}_{nested_key}"

                flat_dict[key] = nested_value
        else:
            # Add the field directly if it's not a nested model
            if field_name in flat_dict:
                if prefix_nested:
                    # Add type prefix if there's a collision
                    field_name = f"{type(model).__name__.lower()}_{field_name}"
                else:
                    # Raise error if collisions aren't allowed
                    raise ValueError(
                        f"Field name collision detected: '{field_name}' appears multiple times. "
                        f"Set prefix_nested=True to resolve."
                    )
            flat_dict[field_name] = field_value

    return flat_dict


def flat_dict_to_pydantic(
    model_cls: Type[PydanticType], flat_dict: dict[str, Any], prefix_nested: bool = False
) -> PydanticType:
    """Simplified version that requires ConfigDict(extra='ignore')"""
    # Check if model has the required ConfigDict setting
    model_config = getattr(model_cls, "model_config", {})
    if model_config.get("extra") != "ignore":
        raise ValueError(
            f"Model {model_cls.__name__} must have ConfigDict(extra='ignore') set. "
            "Add 'model_config = ConfigDict(extra='ignore')' to your model class."
        )

    if not prefix_nested:
        # For prefix_nested=False, group fields by their parent models
        model_data = {}
        for field_name, field_info in model_cls.model_fields.items():
            field_type = field_info.annotation
            if hasattr(field_type, "model_fields"):
                # For nested models, collect all fields that belong to this model
                nested_dict = {}
                for nested_field in field_type.model_fields:
                    if nested_field in flat_dict:
                        nested_dict[nested_field] = flat_dict[nested_field]
                if nested_dict:
                    model_data[field_name] = field_type(**nested_dict)
            else:
                if field_name in flat_dict:
                    model_data[field_name] = flat_dict[field_name]
        return model_cls(**model_data)

    # ... rest of the function for prefix_nested=True remains unchanged ...
    model_data = {}
    for field_name, field_info in model_cls.model_fields.items():
        field_type = field_info.annotation

        # Handle nested Pydantic models
        if hasattr(field_type, "model_fields"):
            # Check nested model's config as well
            nested_config = getattr(field_type, "model_config", {})
            if nested_config.get("extra") != "ignore":
                raise ValueError(
                    f"Nested model {field_type.__name__} must have ConfigDict(extra='ignore') set. "
                    "Add 'model_config = ConfigDict(extra='ignore')' to your model class."
                )

            # Find all keys with this field's prefix
            prefix = f"{field_name}_"
            nested_dict = {k[len(prefix) :]: v for k, v in flat_dict.items() if k.startswith(prefix)}
            if nested_dict:
                model_data[field_name] = flat_dict_to_pydantic(field_type, nested_dict, True)
        else:
            # For non-nested fields, let Pydantic handle it
            if field_name in flat_dict:
                model_data[field_name] = flat_dict[field_name]

    return model_cls(**model_data)


def check_pydantic_model_compatibility(model_cls: Type[PydanticType]) -> tuple[bool, list[str]]:
    """Simplified compatibility check for models using ConfigDict(extra='ignore')"""
    issues = []

    # Check if it's a Pydantic model
    if not inspect.isclass(model_cls) or not issubclass(model_cls, pydantic.BaseModel):
        issues.append(f"Input is not a Pydantic model class: {model_cls}")
        return False, issues

    # Check ConfigDict setting
    model_config = getattr(model_cls, "model_config", {})
    if model_config.get("extra") != "ignore":
        issues.append(f"Model {model_cls.__name__} must have ConfigDict(extra='ignore') set")

    # Check for field aliases
    for field_name, field_info in model_cls.model_fields.items():
        if field_info.alias and field_info.alias != field_name:
            issues.append(f"Field with alias detected: {field_name} -> {field_info.alias}")

        # Check nested models
        field_type = field_info.annotation
        if hasattr(field_type, "model_fields"):
            nested_config = getattr(field_type, "model_config", {})
            if nested_config.get("extra") != "ignore":
                issues.append(f"Nested model {field_type.__name__} must have ConfigDict(extra='ignore') set")

    return len(issues) == 0, issues


if __name__ == "__main__":
    from functools import cached_property

    import pydantic

    from pydantic_to_flat_dict_and_back.src import (
        check_pydantic_model_compatibility,
        flat_dict_to_pydantic,
        pydantic_to_flat_dict,
    )

    class GenerateSignalParams(pydantic.BaseModel):
        T_celsius: float
        generation_frequency: int
        T_total: float
        diameter: float
        lambda_laser: float
        n_medium: float
        variance_fluctuations: float
        offset: float
        seed: int

        model_config = pydantic.ConfigDict(extra="ignore", frozen=True)

    class AnalogToDigitalConverter(pydantic.BaseModel):
        v_min: float
        v_max: float
        n_levels: int  # example 2**16
        sampling_frequency: int  # example 50_000 kHz
        model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

        @cached_property
        def delta_v(self) -> float:
            return (self.v_max - self.v_min) / self.n_levels

    class ParamsAcquisition(pydantic.BaseModel):
        params_gen: GenerateSignalParams
        adc: AnalogToDigitalConverter

        model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

    flat_dict = {
        "uid_gen_sig": "2f9ca4e6401715a9e181a6c92a3cfc9de8084206376eeda5eedd8d1fbf0b4ecf",
        "decay_rate_hetero_theo": 1120.9329117430084,
        "T_celsius": 16.0,
        "generation_frequency": 1000000,
        "T_total": 1.0,
        "diameter": 1e-07,
        "lambda_laser": 9.76e-07,
        "n_medium": 1.33,
        "variance_fluctuations": 1.0,
        "offset": 0.0,
        "seed": 1,
        "v_min": -0.1,
        "v_max": 0.1,
        "n_levels": 65536,
        "sampling_frequency": 50000,
    }

    params_acquisition = flat_dict_to_pydantic(ParamsAcquisition, flat_dict, prefix_nested=False)

    print(params_acquisition)
