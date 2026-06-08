from typing import Any, Dict
import json
import numpy as np
from pathlib import Path


_BASIS_MAP = {"X": 0, "Y": 1, "Z": 2, "I": 3}
_BASIS_INV_MAP = {v: k for k, v in _BASIS_MAP.items()}


#### SAVING AND LOADING STATE ####

def save_state_npz(file_path: Path, amplitudes: np.ndarray, headers: dict[str, dict]) -> None:

    compact_amplitudes = amplitudes.astype(np.complex64, copy=False)
    header_tuples = {name: json.dumps(payload) for name, payload in headers.items()}

    np.savez_compressed(file_path, amplitudes=compact_amplitudes, **header_tuples)


def load_state_npz(file_path: Path) -> tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    with np.load(file_path) as data:
        amplitudes = data["amplitudes"].astype(np.complex128, copy=False)

        headers: Dict[str, Dict[str, Any]] = {}
        for key in data.files:
            if key == "amplitudes":
                continue
            header_json = data[key].item()
            headers[key] = json.loads(header_json)

    return amplitudes, headers


#### SAVING AND LOADING MEASUREMENTS ####


def save_measurements_npz(file_path: Path, values: np.ndarray, bases: list[str], headers: dict[str, dict]) -> None:

    packed_values = np.packbits(np.asarray(values, dtype=np.uint8), axis=1)
    encoded_bases = np.array([_BASIS_MAP[b] for b in bases], dtype=np.uint8) # only for homogen basis
    header_tuples = {name: json.dumps(payload) for name, payload in headers.items()}

    np.savez_compressed(file_path, values=packed_values, bases=encoded_bases, **header_tuples)


def load_measurements_npz(file_path: Path) -> tuple[np.ndarray, list[str], Dict[str, Dict[str, Any]]]:
    with np.load(file_path) as data:
        packed_values = data["values"]
        encoded_bases = data["bases"]

        unpacked = np.unpackbits(packed_values, axis=1)
        n_qubits = encoded_bases.shape[0]
        values = unpacked[:, :n_qubits].astype(np.uint8, copy=False)

        bases = [_BASIS_INV_MAP[int(b)] for b in encoded_bases]

        headers: Dict[str, Dict[str, Any]] = {}
        for key in data.files:
            if key in ("values", "bases"):
                continue
            header_json = data[key].item()
            headers[key] = json.loads(header_json)

    return values, bases, headers
