from typing import Any, Dict
import re
import numpy as np
from pathlib import Path


_INT_RGX = re.compile(r'^[+-]?\d+\Z')
_FLOAT_RGX = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')


#### HEADER FORMATTING AND PARSING ####


def _format_header_txt(name: str, field_dict: dict) -> str:
    label = name.upper()
    parts = [label]
    for key, value in field_dict.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")

    header = " | ".join(parts)
    return header


def _parse_header_txt(header: str) -> tuple[str, Dict[str, Any]]:
    name_str, fields_str = header.split("|", 1)
    name = name_str.strip().lower()

    field_dict = {}
    for field_str in fields_str.split("|"):
        key_str, value_str = field_str.split("=", 1)
        key = key_str.strip()
        value = value_str.strip()

        if _INT_RGX.match(value):
            field_dict[key] = int(value)
        elif _FLOAT_RGX.match(value):
            field_dict[key] = float(value)
        else:
            field_dict[key] = value

    return name, field_dict


#### SAVING AND LOADING STATE ####


def save_state_txt(file_path: Path, amplitudes: np.ndarray, headers: dict[str, dict]) -> None:
    with open(file_path, "w") as f:
        for name, fields in headers.items():
            f.write(_format_header_txt(name, fields) + "\n")

        for c in amplitudes:
            re = float(np.real(c))
            im = float(np.imag(c))
            f.write(f"{re:.10f} {im:.10f}\n")


def load_state_txt(file_path: Path) -> tuple[np.ndarray, Dict[str, Dict[str, Any]]]:

    with open(file_path, "r") as f:
        state_header_name, state_header_fields = _parse_header_txt(f.readline())

        data = np.loadtxt(f, dtype=float)
        if data.ndim == 1:
            data = data[None, :]

    headers = {state_header_name: state_header_fields}
    amplitudes = (data[:, 0] + 1j * data[:, 1]).astype(np.complex128)
    return amplitudes, headers


#### SAVING AND LOADING MEASUREMENTS ####


def save_measurements_txt(file_path: Path, values: np.ndarray, bases: list[str], headers: dict[str, dict]) -> None:
    values = np.asarray(values, dtype=np.uint8)
    with open(file_path, "w") as f:
        for name, fields in headers.items():
            f.write(_format_header_txt(name, fields) + "\n")

        for row in values:
            # currently we broadcast the same basis for all measurements
            encoded_measurements = [op.upper() if bit == 0 else op.lower() for bit, op in zip(row, bases)]
            f.write("".join(encoded_measurements) + "\n")


def load_measurements_txt(file_path: Path) -> tuple[np.ndarray, list[str], Dict[str, Dict[str, Any]]]:
    with open(file_path, "r") as f:
        state_header_name, state_header_fields = _parse_header_txt(f.readline())
        meas_header_name, meas_header_fields = _parse_header_txt(f.readline())

        measurements = [ln.strip() for ln in f]

    headers =  {state_header_name: state_header_fields, meas_header_name: meas_header_fields}

    # assume all measurements use the same basis
    bases = [ch.upper() for ch in measurements[0]]
    n = len(bases)
    m = len(measurements)

    values = np.empty((m, n), dtype=np.uint8)
    for i, s in enumerate(measurements):
        values[i] = [0 if ch.isupper() else 1 for ch in s]

    return values, bases, headers

