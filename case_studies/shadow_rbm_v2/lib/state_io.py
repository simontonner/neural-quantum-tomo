from pathlib import Path
from typing import Dict
import jax.numpy as jnp

def save_state_vector(state_vec: jnp.ndarray, file_path: Path):
    num_qubits = int(jnp.log2(state_vec.shape[0]))

    with open(file_path, "w") as file:
        for idx in range(state_vec.shape[0]):
            amp = state_vec[idx]
            re, im = jnp.real(amp).item(), jnp.imag(amp).item()

            file.write(f"{idx:0{num_qubits}b}: {re:+.8f} {im:+.8f}j\n")

    print(f"State vector written to {file_path} ({file_path.stat().st_size} bytes)")


def load_state_vector(file_path: Path) -> Dict[str, complex]:
    with open(file_path, "r") as file:
        lines = file.readlines()

    state_vec_dict = {}
    for line in lines:
        bitstring, value_str = line.strip().split(": ")

        re_str, im_str = value_str.split()
        re = float(re_str)
        im = float(im_str[:-1])  # remove trailing 'j'
        amplitude = complex(re, im)

        state_vec_dict[bitstring] = amplitude

    return  state_vec_dict


