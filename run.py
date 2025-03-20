# %%
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

# %%
