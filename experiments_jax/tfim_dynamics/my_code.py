##### GENERATION CODE


# =========================
# General utilities
# =========================

def int_to_bitstring(indices: jnp.ndarray, num_bits: int) -> jnp.ndarray:
    indices = indices.astype(jnp.int32)
    powers = 2 ** jnp.arange(num_bits - 1, -1, -1, dtype=jnp.int32)
    bits = (indices[..., None] & powers) > 0
    return bits.astype(jnp.uint8)

def bitstring_to_filestring(bitstring: jnp.ndarray, measurement_basis: List[str]) -> str:
    out = []
    for bit, op in zip([int(b) for b in bitstring], measurement_basis):
        out.append(op.upper() if bit == 0 else op.lower() if bit == 1 else '?')
    return ''.join(out)

def save_state_vector_columns(state: jnp.ndarray, file_path: str) -> None:
    with open(file_path, "w") as f:
        for c in state:
            re = float(jnp.real(c))
            im = float(jnp.imag(c))
            f.write(f"{re:.10f} {im:.10f}\n")

def format_bytes(num: int) -> str:
    n = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


# =========================
# Pauli primitives (inline) — same convention as your W-state file
# =========================

@dataclass
class PauliMeasurement:
    eigenvectors: jnp.ndarray  # shape (2, 2); columns are eigenvectors for outcomes {0,1}

pauli_z = PauliMeasurement(  # columns: |0>, |1>
    eigenvectors=jnp.array([[1.0, 0.0],
                            [0.0, 1.0]], dtype=jnp.complex64)
)
pauli_x = PauliMeasurement(
    eigenvectors=(1 / jnp.sqrt(2))
                 * jnp.array([[1.0, 1.0],
                              [1.0, -1.0]], dtype=jnp.complex64)
)
pauli_y = PauliMeasurement(
    eigenvectors=(1 / jnp.sqrt(2))
                 * jnp.array([[1.0, 1.0],
                              [1.0j, -1.0j]], dtype=jnp.complex64)
)
pauli_i = pauli_z

PAULI_MAP = {"X": pauli_x, "Y": pauli_y, "Z": pauli_z, "I": pauli_i}


# =========================
# MultiQubitMeasurement (tqdm enabled)
# =========================

class MultiQubitMeasurement:
    def __init__(self, meas_dirs: List[str]):
        self.meas_dirs = meas_dirs
        self.pauli_measurements: List[PauliMeasurement] = [PAULI_MAP[c] for c in meas_dirs]
        self.basis_vecs: List[jnp.ndarray] = self._construct_measurement_basis()

    def _construct_measurement_basis(self) -> List[jnp.ndarray]:
        measurement_basis_vectors = []
        outcome_bitstrings = list(product([0, 1], repeat=len(self.pauli_measurements)))
        meas_dirs_str = ''.join(self.meas_dirs)
        for outcome_bitstring in tqdm(outcome_bitstrings, desc=f"Constructing basis {meas_dirs_str}", disable=False):
            multi_qubit_eigenvector = None
            for pauli_measurement, outcome_bit in zip(self.pauli_measurements, outcome_bitstring):
                single_qubit_vector = pauli_measurement.eigenvectors[:, outcome_bit]
                multi_qubit_eigenvector = (
                    single_qubit_vector if multi_qubit_eigenvector is None
                    else jnp.kron(multi_qubit_eigenvector, single_qubit_vector)
                )
            measurement_basis_vectors.append(multi_qubit_eigenvector)
        return measurement_basis_vectors

    def sample_state(self, state_vec: jnp.ndarray, num_samples: int = 1000, rng: PRNGKey | None = None) -> jnp.ndarray:
        rng = PRNGKey(0) if rng is None else rng
        probs = jnp.array([jnp.abs(jnp.vdot(v, state_vec))**2 for v in self.basis_vecs])
        probs /= jnp.sum(probs)
        chosen_indices = jax.random.choice(rng, a=probs.shape[0], shape=(num_samples,), p=probs)
        bitstrings = int_to_bitstring(chosen_indices, len(self.meas_dirs))
        return bitstrings


    for h in h_values:
        print(f"=== [h={h:+.4f}] Build & diagonalize TFIM (NetKet, LxL, PBC) ===")
        hilbert, graph, H = build_tfim(L=side_length, h=h, J=J, pbc=True)
        e0, psi = calculate_groundstate(H)

        # Save state vector (Re, Im) without 'state' in filename
        state_path = out_states / f"tfim_h{h:.2f}_{side_length}x{side_length}.txt"
        save_state_vector_columns(psi, str(state_path))
        saved_states += 1
        print(f"Saved ground-state amplitudes to {state_path}")

        # Magnetization
        mz_total, mz_site = magnetization_z(hilbert, psi)
        print(f"⟨∑ σ^z⟩ = {mz_total:+.6f}  |  per-site ⟨σ^z⟩ = {mz_site:+.6f}")

        # Construct measurement basis (Z^N) with visible tqdm
        basis = ['Z'] * graph.n_nodes
        _ = MultiQubitMeasurement(basis)  # tqdm prints: Constructing basis ZZZ...
        # We don't retain it; re-instantiate below to keep tqdm visible *then* sample
        measurement = MultiQubitMeasurement(basis)

        # Sample all shots in one go (no batch logs)
        rng_master, rng_h = random.split(rng_master)
        samples = measurement.sample_state(psi, num_samples=shots, rng=rng_h)

        meas_path = out_meas / f"tfim_h{h:.2f}_{side_length}x{side_length}_{shots}.txt"
        with open(meas_path, "w") as f:
            for s in samples:
                f.write(bitstring_to_filestring(s, basis) + "\n")
        saved_files += 1
        print(f"Wrote {shots} measurements to {meas_path}")

        # Per-h concise summary
        print(
            "SUMMARY | "
            f"h={h:+.6f} | L={side_length} N={graph.n_nodes} | J={J:+.6f} | "
            f"shots={shots} | E0={e0:.8f} | Mz_total={mz_total:+.6f} Mz_site={mz_site:+.6f}\n"
        )


##### MY DATA LOADING

class MixedDataLoader:
    def __init__(self,
                 bits: jnp.ndarray,
                 field: jnp.ndarray,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 rng: PRNGKey = None):
        """
        bits:  shape (N, num_qubits), dtype uint8
        field: shape (N,),            dtype float32
        """
        assert bits.shape[0] == field.shape[0], "Mismatched sample counts"
        self.bits         = bits
        self.field        = field
        self.dataset_size = bits.shape[0]
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.drop_last    = drop_last
        self.rng          = PRNGKey(0) if rng is None else rng

        # precompute slice boundaries
        self.slice_boundaries = [
            (i, i + batch_size)
            for i in range(0, self.dataset_size, batch_size)
            if not drop_last or (i + batch_size) <= self.dataset_size
        ]

    def __iter__(self):
        self.rng, rng_shuffle = split(self.rng)
        if self.shuffle:
            self.order = permutation(rng_shuffle, self.dataset_size)
        else:
            self.order = jnp.arange(self.dataset_size)
        self.slice_idx = 0
        return self

    def __next__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.slice_idx >= len(self.slice_boundaries):
            raise StopIteration
        start, end = self.slice_boundaries[self.slice_idx]
        self.slice_idx += 1
        idxs = self.order[start:end]
        batch_bits  = self.bits[idxs]   # (batch_size, num_qubits)
        batch_field = self.field[idxs]  # (batch_size,)
        return batch_bits, batch_field

    def __len__(self) -> int:
        return len(self.slice_boundaries)


def load_measurements(file_path: Path) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reads a file where each non-empty line is:
        010110 2.0
    and returns two arrays:
      - bits:  shape (N, num_qubits), dtype uint8
      - field: shape (N,),            dtype float32
    """
    with open(file_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    bits_list: list[jnp.ndarray]  = []
    field_list: list[jnp.ndarray] = []

    for ln in tqdm(lines, desc="Parsing measurements"):
        b, h = filestring_to_sample_tensor(ln)
        bits_list.append(b)
        field_list.append(h)

    bits  = jnp.stack(bits_list,  axis=0)
    field = jnp.stack(field_list, axis=0)
    return bits, field


#### demo old data in file:

# 111111111 1.0
# 111111111 1.0
# 111111111 1.0
# 111101111 1.0
# 000000000 1.0
# 111111111 1.0
# 000000000 1.0
# 000000000 1.0
# 000000000 1.0


#### MY TRAINING LOOP

@jax.jit
def train_step(
        state: TrainState,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
        aux_vars: Dict[str, Any]) -> Tuple[TrainState, jnp.ndarray, Dict[str, Any]]:

    loss_fn = lambda params: state.apply_fn({'params': params}, batch, aux_vars)
    (loss, aux_vars), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, aux_vars


def train(
        state: TrainState,
        loader: MixedDataLoader,
        num_epochs: int,
        key: PRNGKey,
        l2_strength: float,
        lr_schedule_fn: Callable[[int], float]) -> Tuple[TrainState, Dict[int, float]]:

    metrics: Dict[int, Any] = {}
    gibbs_chain = None

    for epoch in range(num_epochs):
        tot_loss = 0.0

        for batch_bits, batch_field in loader:
            if gibbs_chain is None:
                gibbs_chain = jax.random.bernoulli(key, 0.5, shape=batch_bits.shape).astype(jnp.float32)

            key, subkey = jax.random.split(key)
            aux_vars = {
                "key": subkey,
                "l2_strength": l2_strength,
            }

            batch = (batch_bits, batch_field)

            state, loss, aux_vars = train_step(state, batch, aux_vars)
            key = aux_vars["key"]
            free_energy_mean = aux_vars["free_energy_mean"]
            free_energy_std = aux_vars["free_energy_std"]

            tot_loss += float(loss)

        avg_loss = tot_loss / len(loader)
        lr = lr_schedule_fn(state.step)

        metrics[epoch] = dict(
            loss=avg_loss,
            free_energy_mean=free_energy_mean,
            free_energy_var=free_energy_std,
            lr=lr
        )

        print(f"Epoch {epoch+1}/{num_epochs} │ "
              f"Loss: {avg_loss:+.4f} │ "
              f"Free En. Mean: {free_energy_mean:.4f} │ "
              f"Free En. STD: {free_energy_std:.4f} │ "
              f"Learning Rate: {lr:.5f}")

    return state, metrics


file_names = [
    "tfim_h1.00_3x3_10000.txt",
    "tfim_h2.00_3x3_10000.txt",
    "tfim_h2.80_3x3_10000.txt",
    "tfim_h3.00_3x3_10000.txt",
    "tfim_h3.30_3x3_10000.txt",
    "tfim_h3.60_3x3_10000.txt",
    "tfim_h4.00_3x3_10000.txt",
    "tfim_h5.00_3x3_10000.txt",
    "tfim_h6.00_3x3_10000.txt",
    "tfim_h7.00_3x3_10000.txt",
]

bits_list = []
fields_list = []
for fn in file_names:
    bits, field = load_measurements(data_dir / fn)
    bits_list.append(bits)   # (N, num_qubits)
    fields_list.append(field) # (N,)

bits  = jnp.concatenate(bits_list,  axis=0)  # (2*N, num_qubits)
fields = jnp.concatenate(fields_list, axis=0)


batch_size        = 1024
num_visible       = 9
num_hidden        = 9
conditioner_width = 32
num_epochs        = 100
k_steps           = 100
init_lr           = 1e-2
final_lr          = init_lr * 0.1
l2_strength       = 0.0

key = jax.random.PRNGKey(0)
key, key_params, key_dummy = jax.random.split(key, 3)

model = ConditionalRBM(
    num_visible=num_visible,
    num_hidden=num_hidden,
    conditioner_width=conditioner_width,
    k=k_steps
)

# init model with dummy data
bits_dummy   = jnp.zeros((batch_size, num_visible), dtype=jnp.float32)
fields_dummy = jnp.zeros((batch_size,), dtype=jnp.float32)
batch_dummy  = (bits_dummy, fields_dummy)
aux_vars_dummy = { "key": key_dummy, "l2_strength": l2_strength }

variables = model.init({"params": key_params}, batch_dummy, aux_vars_dummy)

loader = MixedDataLoader(bits=bits, field=fields, batch_size=batch_size)

schedule_steps = num_epochs * len(loader)
lr_schedule_fn = get_sigmoid_curve(high=init_lr, low=final_lr, steps=schedule_steps, falloff=0.0005)
optim = optax.adam(lr_schedule_fn)

state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optim)

state, metrics = train(state, loader, num_epochs, key, l2_strength, lr_schedule_fn)


#### NEW DATA IS FORMATTED LIKE (measurement direction in letter and outcome in case):

# XxZZ
# xxzZ
# XXZz
# xXZZ
# XxzZ
# XxZz
# xxZz
# XxzZ
# XxZZ


