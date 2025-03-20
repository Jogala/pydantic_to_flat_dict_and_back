# %%
import pydantic

from pydantic_to_flat_dict_and_back.src import (
    check_pydantic_model_compatibility,
    flat_dict_to_pydantic,
    pydantic_to_flat_dict,
)


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


class C(pydantic.BaseModel):
    value: float
    name: str
    b: B

    model_config = pydantic.ConfigDict(extra="ignore")


c = C(value=1.0, name="test", b=B(value=2.0, name="nested", a=A(value=3.0, name="nested2", seed=42)))

check_pydantic_model_compatibility(C)

flat_d_of_c = pydantic_to_flat_dict(c, prefix_nested=True)
c_from_flat_d = flat_dict_to_pydantic(C, flat_d_of_c, prefix_nested=True)
assert c.model_dump() == c_from_flat_d.model_dump()
# %%
