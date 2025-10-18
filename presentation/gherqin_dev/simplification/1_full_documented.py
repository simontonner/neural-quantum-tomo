import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.convert_parameters import _check_param_device
from torch.nn.utils import parameters_to_vector

import numpy as np


import abc
from collections.abc import MutableSequence

from itertools import chain
from math import ceil

from tqdm import tqdm, tqdm_notebook

import warnings
from functools import wraps

import time




#### COMPLEX OPERATORS


I = torch.Tensor([0, 1])  # noqa: E741


def make_complex(x, y=None):
    """A function that creates a torch compatible complex tensor.

    .. note:: `x` and `y` must have the same shape.

    :param x: The real part or a complex numpy array. If a numpy array,
              will ignore `y`.
    :type x: torch.Tensor or numpy.ndarray
    :param y: The imaginary part. Can be `None`, in which case, the resulting
              complex tensor will have imaginary part equal to zero.
    :type y: torch.Tensor

    :returns: The tensor :math:`[x,y] = x + iy`.
    :rtype: torch.Tensor
    """
    if isinstance(x, np.ndarray):
        return make_complex(torch.tensor(x.real), torch.tensor(x.imag)).contiguous()

    if y is None:
        y = torch.zeros_like(x)
    return torch.cat((x.unsqueeze(0), y.unsqueeze(0)), dim=0)


def numpy(x):
    """Converts a complex torch tensor into a numpy array


    :param x: The tensor to convert.
    :type x: torch.Tensor

    :returns: A complex numpy array containing the data from `x`.
    :rtype: numpy.ndarray
    """
    return real(x).detach().cpu().numpy() + 1j * imag(x).detach().cpu().numpy()


def real(x):
    """Returns the real part of a complex tensor.

    :param x: The complex tensor
    :type x: torch.Tensor

    :returns: The real part of `x`; will have one less dimension than `x`.
    :rtype: torch.Tensor
    """
    return x[0, ...]


def imag(x):
    """Returns the imaginary part of a complex tensor.

    :param x: The complex tensor
    :type x: torch.Tensor

    :returns: The imaginary part of `x`; will have one less dimension than `x`.
    :rtype: torch.Tensor
    """
    return x[1, ...]


def scalar_mult(x, y, out=None):
    """A function that computes the product between complex matrices and scalars,
    complex vectors and scalars or two complex scalars.

    :param x: A complex scalar, vector or matrix.
    :type x: torch.Tensor
    :param y: A complex scalar, vector or matrix.
    :type y: torch.Tensor

    :returns: The product between `x` and `y`.
              Either overwrites `out`, or returns a new tensor.
    :rtype: torch.Tensor
    """
    y = y.to(x)
    if out is None:
        out = torch.zeros(2, *((real(x) * real(y)).shape)).to(x)
    else:
        if out is x or out is y:
            raise RuntimeError("Can't overwrite an argument!")

    torch.mul(real(x), real(y), out=real(out)).sub_(torch.mul(imag(x), imag(y)))
    torch.mul(real(x), imag(y), out=imag(out)).add_(torch.mul(imag(x), real(y)))

    return out


def matmul(x, y):
    """A function that computes complex matrix-matrix and matrix-vector products.

    .. note:: If one wishes to do matrix-vector products, the vector must be
              the second argument (y).

    :param x: A complex matrix.
    :type x: torch.Tensor
    :param y: A complex vector or matrix.
    :type y: torch.Tensor

    :returns: The product between x and y.
    :rtype: torch.Tensor
    """
    y = y.to(x)
    re = torch.matmul(real(x), real(y)).sub_(torch.matmul(imag(x), imag(y)))
    im = torch.matmul(real(x), imag(y)).add_(torch.matmul(imag(x), real(y)))

    return make_complex(re, im)


def inner_prod(x, y):
    """A function that returns the inner product of two complex vectors,
    x and y (<x|y>).

    :param x: A complex vector.
    :type x: torch.Tensor
    :param y: A complex vector.
    :type y: torch.Tensor

    :raises ValueError: If x and y are not complex vectors with their first
                        dimensions being 2, then the function will not execute.

    :returns: The inner product, :math:`\\langle x\\vert y\\rangle`.
    :rtype: torch.Tensor
    """
    y = y.to(x)

    if x.dim() == 2 and y.dim() == 2:
        return make_complex(
            torch.dot(real(x), real(y)) + torch.dot(imag(x), imag(y)),
            torch.dot(real(x), imag(y)) - torch.dot(imag(x), real(y)),
            )
    elif x.dim() == 1 and y.dim() == 1:
        return make_complex(
            (real(x) * real(y)) + (imag(x) * imag(y)),
            (real(x) * imag(y)) - (imag(x) * real(y)),
            )
    else:
        raise ValueError("Unsupported input shapes!")


def outer_prod(x, y):
    """A function that returns the outer product of two complex vectors, x
    and y.

    :param x: A complex vector.
    :type x: torch.Tensor
    :param y: A complex vector.
    :type y: torch.Tensor

    :raises ValueError:	If x and y are not complex vectors with their first
                        dimensions being 2, then an error will be raised.

    :returns: The outer product between x and y,
        :math:`\\vert x \\rangle\\langle y\\vert`.
    :rtype: torch.Tensor
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("An input is not of the right dimension.")

    z = torch.zeros(2, x.size()[1], y.size()[1], dtype=x.dtype, device=x.device)
    z[0] = torch.ger(real(x), real(y)) - torch.ger(imag(x), -imag(y))
    z[1] = torch.ger(real(x), -imag(y)) + torch.ger(imag(x), real(y))

    return z


def einsum(equation, a, b, real_part=True, imag_part=True):
    """Complex-aware version of `einsum`. See the torch documentation for more
    details.

    :param equation: The index equation. Passed directly to `torch.einsum`.
    :type equation: str
    :param a: A complex tensor.
    :type a: torch.Tensor
    :param b: A complex tensor.
    :type b: torch.Tensor
    :param real_part: Whether to compute and return the real part of the result.
    :type real_part: bool
    :param imag_part: Whether to compute and return the imaginary part of the result.
    :type imag_part: bool

    :returns: The Einstein summation of the input tensors performed according to
              the given equation. If both `real_part` and `imag_part` are true,
              the result will be a complex tensor, otherwise a real tensor.
    :rtype: torch.Tensor
    """
    if real_part:
        r = torch.einsum(equation, real(a), real(b)).sub_(
            torch.einsum(equation, imag(a), imag(b))
        )
    if imag_part:
        i = torch.einsum(equation, real(a), imag(b)).add_(
            torch.einsum(equation, imag(a), real(b))
        )

    if real_part and imag_part:
        return make_complex(r, i)
    elif real_part:
        return r
    elif imag_part:
        return i
    else:
        return None


def conjugate(x):
    """Returns the conjugate transpose of the argument.

    In the case of a scalar or vector, only the complex conjugate is taken.
    In the case of a rank-2 or higher tensor, the complex conjugate is taken,
    then the first two indices of the tensor are swapped.

    :param x: A complex tensor.
    :type x: torch.Tensor

    :returns: The conjugate of x.
    :rtype: torch.Tensor
    """
    if x.dim() < 3:
        return conj(x)
    else:
        return make_complex(
            torch.transpose(real(x), 0, 1), -torch.transpose(imag(x), 0, 1)
        )


def conj(x):
    """Returns the element-wise complex conjugate of the argument.

    :param x: A complex tensor.
    :type x: torch.Tensor

    :returns: The complex conjugate of x.
    :rtype: torch.Tensor
    """
    return make_complex(real(x), -imag(x))


def elementwise_mult(x, y):
    """Alias for :func:`scalar_mult`."""
    return scalar_mult(x, y)


def elementwise_division(x, y):
    """Element-wise division of x by y.

    :param x: A complex tensor.
    :type x: torch.Tensor
    :param y: A complex tensor.
    :type y: torch.Tensor

    :rtype: torch.Tensor
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape!")

    y_star = conj(y)

    sqrd_abs_y = absolute_value(y).pow_(2)

    return elementwise_mult(x, y_star).div_(sqrd_abs_y)


def absolute_value(x):
    """Returns the complex absolute value elementwise.

    :param x: A complex tensor.
    :type x: torch.Tensor

    :returns: A real tensor.
    :rtype: torch.Tensor
    """
    x_star = conj(x)
    return real(elementwise_mult(x, x_star)).sqrt_()


def kronecker_prod(x, y):
    """Returns the tensor / Kronecker product of 2 complex matrices, x and y.

    :param x: A complex matrix.
    :type x: torch.Tensor
    :param y: A complex matrix.
    :type y: torch.Tensor

    :raises ValueError: If x and y do not have 3 dimensions or their first
                        dimension is not 2, the function cannot execute.

    :returns: The Kronecker product of x and y, :math:`x \\otimes y`.
    :rtype: torch.Tensor
    """
    if not (x.dim() == y.dim() == 3):
        raise ValueError("Inputs must be complex matrices!")

    return einsum("ab,cd->acbd", x, y).reshape(
        2, x.shape[1] * y.shape[1], x.shape[2] * y.shape[2]
    )


def sigmoid(x, y):
    r"""Returns the sigmoid function of a complex number. Acts elementwise.

    :param x: The real part of the complex number
    :type x: torch.Tensor
    :param y: The imaginary part of the complex number
    :type y: torch.Tensor
    :returns: The complex sigmoid of :math:`x + iy`
    :rtype: torch.Tensor
    """
    z = (x.cpu().numpy()) + 1j * (y.cpu().numpy())

    out = np.exp(z) / (1 + np.exp(z))
    out = torch.tensor([np.real(out), np.imag(out)]).to(x)

    return out


def scalar_divide(x, y):
    """Divides `x` by `y`.
    If `x` and `y` have the same shape, then acts elementwise.
    If `y` is a complex scalar, then performs a scalar division.

    :param x: The numerator (a complex tensor).
    :type x: torch.Tensor
    :param y: The denominator (a complex tensor).
    :type y: torch.Tensor

    :returns: x / y
    :rtype: torch.Tensor
    """
    return scalar_mult(x, inverse(y))


def inverse(z):
    """Returns the multiplicative inverse of `z`. Acts elementwise.

    :param z: The complex tensor.
    :type z: torch.Tensor

    :returns: 1 / z
    :rtype: torch.Tensor
    """
    z_star = conj(z)
    denominator = real(scalar_mult(z, z_star))

    return z_star / denominator


def norm_sqr(x):
    """Returns the squared norm of the argument.

    :param x: A complex scalar.
    :type x: torch.Tensor

    :returns: :math:`|x|^2`.
    :rtype: torch.Tensor
    """
    return real(inner_prod(x, x))


def norm(x):
    """Returns the norm of the argument.

    :param x: A complex scalar.
    :type x: torch.Tensor

    :returns: :math:`|x|`.
    :rtype: torch.Tensor
    """
    return norm_sqr(x).sqrt_()



#### UNITARIES


def create_dict(**kwargs):
    r"""A function that creates a dictionary of unitary operators.

    By default, the dictionary contains the unitaries which perform a change of
    basis from the computational basis (Pauli-Z) to one of the other Pauli
    bases. The default keys (`X`, `Y`, and `Z`) denote the target basis.

    :param \**kwargs: Keyword arguments of any unitary operators to add to the
                      resulting dictionary. The given operators will overwrite
                      the default matrices if they share the same key.

    :returns: A dictionary of unitaries.
    :rtype: dict
    """
    dictionary = {
        "X": torch.tensor(
            [[[1.0, 1.0], [1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=torch.double
        )
             / np.sqrt(2),
        "Y": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[0.0, -1.0], [0.0, 1.0]]], dtype=torch.double
        )
             / np.sqrt(2),
        "Z": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=torch.double
        ),
    }

    dictionary.update(
        {
            name: (
                matrix.clone().detach()
                if isinstance(matrix, torch.Tensor)
                else torch.tensor(matrix)
            ).to(dtype=torch.double)
            for name, matrix in kwargs.items()
        }
    )

    return dictionary


def _kron_mult(matrices, x):
    n = [m.size()[0] for m in matrices]
    l, r = np.prod(n), 1  # noqa: E741

    if l != x.shape[1]:  # noqa: E741
        raise ValueError("Incompatible sizes!")

    y = x.clone()
    for s in reversed(range(len(n))):
        l //= n[s]  # noqa: E741
        m = matrices[s]

        for k in range(l):
            for i in range(r):
                slc = slice(k * n[s] * r + i, (k + 1) * n[s] * r + i, r)
                temp = y[:, slc, ...]
                y[:, slc, ...] = matmul(m, temp)
        r *= n[s]

    return y


def rotate_psi(nn_state, basis, space, unitaries=None, psi=None):
    r"""A function that rotates the reconstructed wavefunction to a different
    basis.

    :param nn_state: The neural network state (i.e. complex wavefunction or
                     positive wavefunction).
    :type nn_state: qucumber.nn_states.WaveFunctionBase
    :param basis: The basis to rotate the wavefunction to.
    :type basis: str
    :param space: The basis elements of the Hilbert space of the system :math:`\mathcal{H}`.
    :type space: torch.Tensor
    :param unitaries: A dictionary of (2x2) unitary operators.
    :type unitaries: dict(str, torch.Tensor)
    :param psi: A wavefunction that the user can input to override the neural
                network state's wavefunction.
    :type psi: torch.Tensor

    :returns: A wavefunction in a new basis.
    :rtype: torch.Tensor
    """
    psi = (
        nn_state.psi(space)
        if psi is None
        else psi.to(dtype=torch.double, device=nn_state.device)
    )

    unitaries = unitaries if unitaries else nn_state.unitary_dict
    unitaries = {k: v.to(device=nn_state.device) for k, v in unitaries.items()}
    us = [unitaries[b] for b in basis]
    return _kron_mult(us, psi)


def rotate_rho(nn_state, basis, space, unitaries=None, rho=None):
    r"""Computes the density matrix rotated into some basis.

    :param nn_state: The density matrix neural network state.
    :type nn_state: qucumber.nn_states.DensityMatrix
    :param basis: The basis to rotate the density matrix to.
    :type basis: str
    :param space: The basis elements of the Hilbert space of the system :math:`\mathcal{H}`.
    :type space: torch.Tensor
    :param unitaries: A dictionary of unitary matrices associated with
                        rotation into each basis
    :type unitaries: dict(str, torch.Tensor)
    :param rho: A density matrix that the user can input to override the neural
                network state's density matrix.
    :type rho: torch.Tensor

    :returns: The rotated density matrix
    :rtype: torch.Tensor
    """
    rho = (
        nn_state.rho(space, space)
        if rho is None
        else rho.to(dtype=torch.double, device=nn_state.device)
    )

    unitaries = unitaries if unitaries else nn_state.unitary_dict
    unitaries = {k: v.to(device=nn_state.device) for k, v in unitaries.items()}
    us = [unitaries[b] for b in basis]

    rho_r = _kron_mult(us, rho)
    rho_r = _kron_mult(us, conjugate(rho_r))

    return rho_r


def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    unitaries = unitaries if unitaries else nn_state.unitary_dict
    unitaries = {k: v.to(device="cpu") for k, v in unitaries.items()}

    basis = np.array(list(basis))
    sites = np.where(basis != "Z")[0]

    if sites.size != 0:
        Us = torch.stack([unitaries[b] for b in basis[sites]]).cpu().numpy()

        reps = [1 for _ in states.shape]
        v = states.unsqueeze(0).repeat(2 ** sites.size, *reps)
        v[..., sites] = nn_state.generate_hilbert_space(size=sites.size).unsqueeze(1)
        v = v.contiguous()

        int_sample = states[..., sites].round().int().cpu().numpy()
        ints_size = np.arange(sites.size)

        # overwrite rotated elements
        int_vp = v[..., sites].long().cpu().numpy()
        all_Us = Us[ints_size, :, int_sample, int_vp]

        Ut = np.prod(all_Us[..., 0] + (1j * all_Us[..., 1]), axis=-1)
    else:
        v = states.unsqueeze(0)
        Ut = np.ones(v.shape[:-1], dtype=complex)

    return Ut, v


def _convert_basis_element_to_index(states):
    powers = (2 ** (torch.arange(states.shape[-1], 0, -1) - 1)).to(states)
    return torch.matmul(states, powers)


def rotate_psi_inner_prod(
        nn_state, basis, states, unitaries=None, psi=None, include_extras=False
):
    r"""A function that rotates the wavefunction to a different
    basis and then computes the resulting amplitude of a batch of basis elements.

    :param nn_state: The neural network state (i.e. complex wavefunction or
                     positive wavefunction).
    :type nn_state: qucumber.nn_states.WaveFunctionBase
    :param basis: The basis to rotate the wavefunction to.
    :type basis: str
    :param states: The batch of basis elements to compute the amplitudes of.
    :type states: torch.Tensor
    :param unitaries: A dictionary of (2x2) unitary operators.
    :type unitaries: dict(str, torch.Tensor)
    :param psi: A wavefunction that the user can input to override the neural
                network state's wavefunction.
    :type psi: torch.Tensor
    :param include_extras: Whether to include all the terms of the summation as
                           well as the expanded basis states in the output.
    :type include_extras: bool

    :returns: Amplitude of the states wrt the rotated wavefunction, and
              possibly some extras.
    :rtype: torch.Tensor or tuple(torch.Tensor)
    """
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)

    if psi is None:
        psi = nn_state.psi(v).detach()
    else:
        # pick out the entries of psi that we actually need
        idx = _convert_basis_element_to_index(v).long()
        psi = psi[:, idx]

    psi = numpy(psi.cpu())
    Ut *= psi

    Upsi_v = make_complex(Ut).to(dtype=torch.double, device=nn_state.device)
    Upsi = torch.sum(Upsi_v, dim=1)

    if include_extras:
        return Upsi, Upsi_v, v
    else:
        return Upsi


def rotate_rho_probs(
        nn_state, basis, states, unitaries=None, rho=None, include_extras=False
):
    r"""A function that rotates the wavefunction to a different
    basis and then computes the resulting Born rule probability of a basis
    element.

    :param nn_state: The density matrix neural network state.
    :type nn_state: qucumber.nn_states.DensityMatrix
    :param basis: The basis to rotate the density matrix to.
    :type basis: str
    :param states: The batch of basis elements to compute the probabilities of.
    :type states: torch.Tensor
    :param unitaries: A dictionary of (2x2) unitary operators.
    :type unitaries: dict(str, torch.Tensor)
    :param rho: A density matrix that the user can input to override the neural
                network state's density matrix.
    :type rho: torch.Tensor
    :param include_extras: Whether to include all the terms of the summation as
                           well as the expanded basis states in the output.
    :type include_extras: bool

    :returns: Probability of the states wrt the rotated density matrix, and
              possibly some extras.
    :rtype: torch.Tensor or tuple(torch.Tensor)
    """
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)
    Ut = np.einsum("ib,jb->ijb", Ut, np.conj(Ut))

    if rho is None:
        rho = nn_state.rho(v).detach()
    else:
        # pick out the entries of rho that we actually need
        idx = _convert_basis_element_to_index(v).long()
        rho = rho[:, idx.unsqueeze(0), idx.unsqueeze(1)]

    rho = numpy(rho.cpu())
    Ut *= rho

    UrhoU_v = make_complex(Ut).to(dtype=torch.double, device=nn_state.device)
    UrhoU = torch.sum(
        real(UrhoU_v), dim=(0, 1)
    )  # imaginary parts will cancel out anyway

    if include_extras:
        return UrhoU, UrhoU_v, v
    else:
        return UrhoU


#### DATA UTILS

def load_data(tr_samples_path, tr_psi_path=None, tr_bases_path=None, bases_path=None):
    r"""Load the data required for training.

    :param tr_samples_path: The path to the training data.
    :type tr_samples_path: str
    :param tr_psi_path: The path to the target/true wavefunction.
    :type tr_psi_path: str
    :param tr_bases_path: The path to the basis data.
    :type tr_bases_path: str
    :param bases_path: The path to a file containing all possible bases used in
                       the tr_bases_path file.
    :type bases_path: str

    :returns: A list of all input parameters.
    :rtype: list
    """
    data = []
    data.append(
        torch.tensor(np.loadtxt(tr_samples_path, dtype="float32"), dtype=torch.double)
    )

    if tr_psi_path is not None:
        target_psi_data = np.loadtxt(tr_psi_path, dtype="float32")
        target_psi = torch.zeros(2, len(target_psi_data), dtype=torch.double)
        target_psi[0] = torch.tensor(target_psi_data[:, 0], dtype=torch.double)
        target_psi[1] = torch.tensor(target_psi_data[:, 1], dtype=torch.double)
        data.append(target_psi)

    if tr_bases_path is not None:
        data.append(np.loadtxt(tr_bases_path, dtype=str))

    if bases_path is not None:
        data.append(np.loadtxt(bases_path, dtype=str, ndmin=1))
    return data


def load_data_DM(
        tr_samples_path,
        tr_mtx_real_path=None,
        tr_mtx_imag_path=None,
        tr_bases_path=None,
        bases_path=None,
):
    r"""Load the data required for training.

    :param tr_samples_path: The path to the training data.
    :type tr_samples_path: str
    :param tr_mtx_real_path: The path to the real part of the density matrix
    :type tr_mtx_real_path: str
    :param tr_mtx_imag_path: The path to the imaginary part of the density matrix
    :type tr_mtx_imag_path: str
    :param tr_bases_path: The path to the basis data.
    :type tr_bases_path: str
    :param bases_path: The path to a file containing all possible bases used in
                       the tr_bases_path file.
    :type bases_path: str

    :returns: A list of all input parameters, with the real and imaginary parts
              of the target density matrix (if provided) combined into one complex matrix.
    :rtype: list
    """
    data = []
    data.append(
        torch.tensor(np.loadtxt(tr_samples_path, dtype="float32"), dtype=torch.double)
    )

    if tr_mtx_real_path is not None:
        mtx_real = torch.tensor(
            np.loadtxt(tr_mtx_real_path, dtype="float32"), dtype=torch.double
        )

    if tr_mtx_imag_path is not None:
        mtx_imag = torch.tensor(
            np.loadtxt(tr_mtx_imag_path, dtype="float32"), dtype=torch.double
        )

    if tr_mtx_real_path is not None or tr_mtx_imag_path is not None:
        if tr_mtx_real_path is None or tr_mtx_imag_path is None:
            raise ValueError("Must provide a real and imaginary part of target matrix!")
        else:
            data.append(make_complex(mtx_real, mtx_imag))

    if tr_bases_path is not None:
        data.append(np.loadtxt(tr_bases_path, dtype=str))

    if bases_path is not None:
        data.append(np.loadtxt(bases_path, dtype=str, ndmin=1))

    return data


def extract_refbasis_samples(train_samples, train_bases):
    r"""Extract the reference basis samples from the data.

    :param train_samples: The training samples.
    :type train_samples: torch.Tensor
    :param train_bases: The bases of the training samples.
    :type train_bases: numpy.ndarray

    :returns: The samples in the data that are only in the reference basis.
    :rtype: torch.Tensor
    """
    torch_ver = int(torch.__version__[:3].replace(".", ""))
    dtype = torch.bool if torch_ver >= 12 else torch.uint8

    idx = (
        torch.tensor((train_bases == "Z").astype(np.uint8))
        .all(dim=1)
        .to(device=train_samples.device, dtype=dtype)
    )
    z_samples = train_samples[idx]
    return z_samples


#### GRADIENT UTILS

# based on code found in PyTorch
def vector_to_grads(vec, parameters):
    r"""Convert one vector to the parameters.

    :param vec: a single vector represents the parameters of a model.
    :type vec:  torch.Tensor
    :param parameters: an iterator of Tensors that are the parameters of a
                       model.
    :type parameters: list[torch.Tensor]
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter gradient
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()

        # Slice the vector, reshape it, and replace the gradient data of
        # the parameter
        param.grad = vec[pointer : pointer + num_param].view(param.size()).data

        # Increment the pointer
        pointer += num_param


#### RBM CLASS


class auto_unsqueeze_args:
    def __init__(self, *arg_indices):
        self.arg_indices = list(arg_indices)

        if len(self.arg_indices) == 0:
            self.arg_indices.append(1)

    def __call__(self, f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            args = list(args)
            unsqueeze = False

            for a in self.arg_indices:
                if args[a].dim() < 2:
                    unsqueeze = True
                    args[a] = args[a].unsqueeze(0)

            if unsqueeze:  # remove superfluous axis, if it exists
                return f(*args, **kwargs).squeeze_(0)
            else:
                return f(*args, **kwargs)

        return wrapped_f



class BinaryRBM(nn.Module):
    def __init__(self, num_visible, num_hidden=None, zero_weights=False, gpu=True):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.num_pars = (
                (self.num_visible * self.num_hidden) + self.num_visible + self.num_hidden
        )

        self.gpu = gpu and torch.cuda.is_available()

        self.device = torch.device("cuda") if self.gpu else torch.device("cpu")

        self.initialize_parameters(zero_weights=zero_weights)

    def __repr__(self):
        return (
            f"BinaryRBM(num_visible={self.num_visible}, "
            f"num_hidden={self.num_hidden}, gpu={self.gpu})"
        )

    def initialize_parameters(self, zero_weights=False):
        """Randomize the parameters of the RBM"""

        gen_tensor = torch.zeros if zero_weights else torch.randn
        self.weights = nn.Parameter(
            (
                    gen_tensor(
                        self.num_hidden,
                        self.num_visible,
                        device=self.device,
                        dtype=torch.double,
                    )
                    / np.sqrt(self.num_visible)
            ),
            requires_grad=False,
        )

        self.visible_bias = nn.Parameter(
            torch.zeros(self.num_visible, device=self.device, dtype=torch.double),
            requires_grad=False,
        )
        self.hidden_bias = nn.Parameter(
            torch.zeros(self.num_hidden, device=self.device, dtype=torch.double),
            requires_grad=False,
        )

    @auto_unsqueeze_args()
    def effective_energy(self, v):
        r"""The effective energies of the given visible states.

        .. math::

            \mathcal{E}(\bm{v}) &= -\sum_{j}b_j v_j
                        - \sum_{i}\log
                            \left\lbrack 1 +
                                  \exp\left(c_{i} + \sum_{j} W_{ij} v_j\right)
                            \right\rbrack

        :param v: The visible states.
        :type v: torch.Tensor

        :returns: The effective energies of the given visible states.
        :rtype: torch.Tensor
        """
        v = v.to(self.weights)
        visible_bias_term = torch.matmul(v, self.visible_bias)
        hid_bias_term = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1)

        return -(visible_bias_term + hid_bias_term)

    def effective_energy_gradient(self, v, reduce=True):
        """The gradients of the effective energies for the given visible states.

        :param v: The visible states.
        :type v: torch.Tensor
        :param reduce: If `True`, will sum over the gradients resulting from
                       each visible state. Otherwise will return a batch of
                       gradient vectors.
        :type reduce: bool

        :returns: Will return a vector (or matrix if `reduce=False` and multiple
                  visible states were given as a matrix) containing the gradients
                  for all parameters (computed on the given visible states v).
        :rtype: torch.Tensor
        """
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.weights)
        prob = self.prob_h_given_v(v)

        if reduce:
            W_grad = -torch.matmul(prob.transpose(0, -1), v)
            vb_grad = -torch.sum(v, 0)
            hb_grad = -torch.sum(prob, 0)
            return parameters_to_vector([W_grad, vb_grad, hb_grad])
        else:
            W_grad = -torch.einsum("...j,...k->...jk", prob, v)
            vb_grad = -v
            hb_grad = -prob
            vec = [W_grad.view(*v.shape[:-1], -1), vb_grad, hb_grad]
            return torch.cat(vec, dim=-1)

    @auto_unsqueeze_args()
    def prob_v_given_h(self, h, out=None):
        """Given a hidden unit configuration, compute the probability
        vector of the visible units being on.

        :param h: The hidden unit
        :type h: torch.Tensor
        :param out: The output tensor to write to.
        :type out: torch.Tensor

        :returns: The probability of visible units being active given the
                  hidden state.
        :rtype: torch.Tensor
        """
        return (
            torch.matmul(h, self.weights.data, out=out)
            .add_(self.visible_bias.data)
            .sigmoid_()
            .clamp_(min=0, max=1)
        )

    @auto_unsqueeze_args()
    def prob_h_given_v(self, v, out=None):
        """Given a visible unit configuration, compute the probability
        vector of the hidden units being on.

        :param h: The hidden unit.
        :type h: torch.Tensor
        :param out: The output tensor to write to.
        :type out: torch.Tensor

        :returns: The probability of hidden units being active given the
                  visible state.
        :rtype: torch.Tensor
        """
        return (
            torch.matmul(v, self.weights.data.t(), out=out)
            .add_(self.hidden_bias.data)
            .sigmoid_()
            .clamp_(min=0, max=1)
        )

    def sample_v_given_h(self, h, out=None):
        """Sample/generate a visible state given a hidden state.

        :param h: The hidden state.
        :type h: torch.Tensor
        :param out: The output tensor to write to.
        :type out: torch.Tensor

        :returns: The sampled visible state.
        :rtype: torch.Tensor
        """
        v = self.prob_v_given_h(h, out=out)
        v = torch.bernoulli(v, out=out)  # overwrite v with its sample
        return v

    def sample_h_given_v(self, v, out=None):
        """Sample/generate a hidden state given a visible state.

        :param h: The visible state.
        :type h: torch.Tensor
        :param out: The output tensor to write to.
        :type out: torch.Tensor

        :returns: The sampled hidden state.
        :rtype: torch.Tensor
        """
        h = self.prob_h_given_v(v, out=out)
        h = torch.bernoulli(h, out=out)  # overwrite h with its sample
        return h

    def gibbs_steps(self, k, initial_state, overwrite=False):
        r"""Performs k steps of Block Gibbs sampling. One step consists of sampling
        the hidden state :math:`\bm{h}` from the conditional distribution
        :math:`p(\bm{h}\:|\:\bm{v})`, and sampling the visible
        state :math:`\bm{v}` from the conditional distribution
        :math:`p(\bm{v}\:|\:\bm{h})`.

        :param k: Number of Block Gibbs steps.
        :type k: int
        :param initial_state: The initial state of the Markov Chains.
        :type initial_state: torch.Tensor
        :param overwrite: Whether to overwrite the initial_state tensor.
                          Exception: If initial_state is not on the same device
                          as the RBM, it will NOT be overwritten.
        :type overwrite: bool

        :returns: Returns the visible states after k steps of
                  Block Gibbs sampling
        :rtype: torch.Tensor
        """
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)

        h = torch.zeros(*v.shape[:-1], self.num_hidden).to(self.weights)

        for _ in range(k):
            self.sample_h_given_v(v, out=h)
            self.sample_v_given_h(h, out=v)

        return v

    def partition(self, space):
        """Compute the partition function of the RBM.

        :param space: A rank 2 tensor of the visible space.
        :type space: torch.Tensor

        :returns: The value of the partition function evaluated at the current
                  state of the RBM.
        :rtype: torch.Tensor
        """
        logZ = (-self.effective_energy(space)).logsumexp(0)
        return logZ.exp()


#### CALLBACKS



class CallbackBase:
    """Base class for callbacks."""

    def on_train_start(self, nn_state):
        """Called at the start of the training cycle.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        """
        pass

    def on_train_end(self, nn_state):
        """Called at the end of the training cycle.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        """
        pass

    def on_epoch_start(self, nn_state, epoch):
        """Called at the start of each epoch.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param epoch: The current epoch.
        :type epoch: int
        """
        pass

    def on_epoch_end(self, nn_state, epoch):
        """Called at the end of each epoch.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param epoch: The current epoch.
        :type epoch: int
        """
        pass

    def on_batch_start(self, nn_state, epoch, batch):
        """Called at the start of each batch.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param epoch: The current epoch.
        :type epoch: int
        :param batch: The current batch index.
        :type batch: int
        """
        pass

    def on_batch_end(self, nn_state, epoch, batch):
        """Called at the end of each batch.

        :param nn_state: The WaveFunction being trained.
        :type nn_state: qucumber.nn_states.WaveFunctionBase
        :param epoch: The current epoch.
        :type epoch: int
        :param batch: The current batch index.
        :type batch: int
        """
        pass




class CallbackList(CallbackBase, MutableSequence):
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = list(callbacks)

    def __len__(self):
        return len(self.callbacks)

    def __getitem__(self, key):
        return self.callbacks[key]

    def __setitem__(self, key, value):
        if isinstance(value, CallbackBase):
            self.callbacks[key] = value
        else:
            raise TypeError(
                "value must be an instance of qucumber.callbacks.CallbackBase"
            )

    def __delitem__(self, index):
        del self.callbacks[index]

    def __iter__(self):
        return iter(self.callbacks)

    def __add__(self, other):
        return CallbackList(self.callbacks + other.callbacks)

    def insert(self, index, value):
        if isinstance(value, CallbackBase):
            self.callbacks.insert(index, value)
        else:
            raise TypeError(
                "value must be an instance of qucumber.callbacks.CallbackBase"
            )

    def on_train_start(self, rbm):
        for cb in self.callbacks:
            cb.on_train_start(rbm)

    def on_train_end(self, rbm):
        for cb in self.callbacks:
            cb.on_train_end(rbm)

    def on_epoch_start(self, rbm, epoch):
        for cb in self.callbacks:
            cb.on_epoch_start(rbm, epoch)

    def on_epoch_end(self, rbm, epoch):
        for cb in self.callbacks:
            cb.on_epoch_end(rbm, epoch)

    def on_batch_start(self, rbm, epoch, batch):
        for cb in self.callbacks:
            cb.on_batch_start(rbm, epoch, batch)

    def on_batch_end(self, rbm, epoch, batch):
        for cb in self.callbacks:
            cb.on_batch_end(rbm, epoch, batch)



class Timer(CallbackBase):
    """Callback which records the training time.

    This callback is always called at the start and end of training. It will
    run at the end of an epoch or batch if the given model's `stop_training`
    property is set to True.

    :param verbose: Whether to print the elapsed time at the end of training.
    :type verbose: bool
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.already_notified = False

    def on_train_start(self, nn_state):
        self.start_time = time.time()

    def on_batch_end(self, nn_state, epoch, batch):
        if nn_state.stop_training:
            if self.verbose and not self.already_notified:
                print(f"Training terminated at epoch: {epoch}, batch: {batch}")
                self.already_notified = True

    def on_epoch_end(self, nn_state, epoch):
        if nn_state.stop_training:
            if self.verbose and not self.already_notified:
                print(f"Training terminated at epoch: {epoch}")
                self.already_notified = True

    def on_train_end(self, nn_state):
        self.calculate_elapsed_time()

    def calculate_elapsed_time(self):
        self.end_time = time.time()
        self.training_time = self.end_time - self.start_time
        if self.verbose:
            print(
                "Total time elapsed during training: {:6.3f} s".format(
                    self.training_time
                )
            )





#### COMPLEX WAVEFUNCTION


class NeuralStateBase(abc.ABC):
    """Abstract Base Class for Neural Network Quantum States."""

    _stop_training = False

    @property
    def stop_training(self):
        """If `True`, will not train.

        If this property is set to `True` during the training cycle, training
        will terminate once the current batch or epoch ends (depending on when
        `stop_training` was set).
        """
        return self._stop_training

    @stop_training.setter
    def stop_training(self, new_val):
        if isinstance(new_val, bool):
            self._stop_training = new_val
        else:
            raise ValueError("stop_training must be a boolean value!")

    @property
    def max_size(self):
        """Maximum size of the Hilbert space for full enumeration"""
        return 20

    @property
    @abc.abstractmethod
    def networks(self):
        """A list of the names of the internal RBMs."""

    @property
    @abc.abstractmethod
    def rbm_am(self):
        """The RBM to be used to learn the wavefunction amplitude."""

    @rbm_am.setter
    @abc.abstractmethod
    def rbm_am(self, new_val):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device(self):
        """The device that the model is on."""

    @device.setter
    @abc.abstractmethod
    def device(self, new_val):
        raise NotImplementedError

    def reinitialize_parameters(self):
        """Randomize the parameters of the internal RBMs."""
        for net in self.networks:
            getattr(self, net).initialize_parameters()

    def __getattr__(self, attr):
        return getattr(self.rbm_am, attr)

    def probability(self, v, Z=1.0):
        """Evaluates the probability of the given vector(s) of visible
        states. Assumes the visible states were measured in the computational
        basis.

        :param v: The visible states.
        :type v: torch.Tensor
        :param Z: The partition function / normalization constant.
                  Defaults to 1, producing unnormalized probabilities.
        :type Z: float

        :returns: The probability of the given vector(s) of visible units.
        :rtype: torch.Tensor
        """
        v = v.to(device=self.device, dtype=torch.double)
        return (-self.rbm_am.effective_energy(v)).exp() / Z

    def sample(self, k, num_samples=1, initial_state=None, overwrite=False):
        r"""Performs k steps of Block Gibbs sampling. One step consists of sampling
        the hidden state :math:`\bm{h}` from the conditional distribution
        :math:`p_{\bm{\lambda}}(\bm{h}\:|\:\bm{v})`, and sampling the visible
        state :math:`\bm{v}` from the conditional distribution
        :math:`p_{\bm{\lambda}}(\bm{v}\:|\:\bm{h})`.

        :param k: Number of Block Gibbs steps.
        :type k: int
        :param num_samples: The number of samples to generate.
        :type num_samples: int
        :param initial_state: The initial state of the Markov Chains. If given,
                              `num_samples` will be ignored.
        :type initial_state: torch.Tensor
        :param overwrite: Whether to overwrite the initial_state tensor, if it is provided.
        :type overwrite: bool
        """
        if initial_state is None:
            dist = torch.distributions.Bernoulli(probs=0.5)
            sample_size = torch.Size((num_samples, self.num_visible))
            initial_state = dist.sample(sample_size).to(
                device=self.device, dtype=torch.double
            )

        return self.rbm_am.gibbs_steps(k, initial_state, overwrite=overwrite)

    def subspace_vector(self, num, size=None, device=None):
        r"""Generates a single vector from the Hilbert space of dimension
        :math:`2^{\text{size}}`.

        :param size: The size of each element of the Hilbert space.
        :type size: int
        :param num: The specific vector to return from the Hilbert space. Since
                    the Hilbert space can be represented by the set of binary strings
                    of length `size`, `num` is equivalent to the decimal representation
                    of the returned vector.
        :type num: int
        :param device: The device to create the vector on. Defaults to the
                       device this model is on.

        :returns: A state from the Hilbert space.
        :rtype: torch.Tensor
        """
        device = device if device is not None else self.device
        size = size if size else self.num_visible
        space = ((num & (1 << np.arange(size))) > 0)[::-1]
        space = space.astype(int)
        return torch.tensor(space, dtype=torch.double, device=device)

    def generate_hilbert_space(self, size=None, device=None):
        r"""Generates Hilbert space of dimension :math:`2^{\text{size}}`.

        :param size: The size of each element of the Hilbert space. Defaults to
                     the number of visible units.
        :type size: int
        :param device: The device to create the Hilbert space matrix on.
                       Defaults to the device this model is on.

        :returns: A tensor with all the basis states of the Hilbert space.
        :rtype: torch.Tensor
        """
        device = device if device is not None else self.device
        size = size if size else self.rbm_am.num_visible
        if size > self.max_size:
            raise ValueError("Size of the Hilbert space is too large!")
        else:
            dim = np.arange(2 ** size)
            space = ((dim[:, None] & (1 << np.arange(size))) > 0)[:, ::-1]
            space = space.astype(int)
            return torch.tensor(space, dtype=torch.double, device=device)

    def normalization(self, space):
        r"""Compute the normalization constant of the state.
        In the case of a pure state, this is the norm of the unnormalized wavefunction.
        In the case of a mixed state, this is the trace of the unnormalized density
        matrix.

        .. math::

            Z_{\bm{\lambda}}=
            \sum_{\bm{\sigma}} p_{\bm{\lambda}}(\bm{\sigma})

        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor

        """
        return self.rbm_am.partition(space)

    def compute_normalization(self, space):
        """Alias for :func:`normalization<qucumber.nn_states.NeuralStateBase.normalization>`"""
        return self.normalization(space)

    def save(self, location, metadata=None):
        """Saves the NeuralState parameters to the given location along with
        any given metadata.

        :param location: The location to save the data.
        :type location: str or file
        :param metadata: Any extra metadata to store alongside the NeuralState
                         parameters.
        :type metadata: dict
        """
        # add extra metadata to dictionary before saving it to disk
        metadata = metadata if metadata else {}

        if hasattr(self, "unitary_dict"):
            if "unitary_dict" in metadata.keys():
                raise ValueError(
                    "Invalid key in metadata; unitary_dict cannot be a key!"
                )
            metadata["unitary_dict"] = self.unitary_dict

        # validate metadata
        for net in self.networks:
            if net in metadata.keys():
                raise ValueError(f"Invalid key in metadata; '{net}' cannot be a key!")

        data = {net: getattr(self, net).state_dict() for net in self.networks}
        data.update(**metadata)
        torch.save(data, location)

    def load(self, location):
        """Loads the NeuralState parameters from the given location ignoring any
        metadata stored in the file. Overwrites the NeuralState's parameters.

        .. note::
            The NeuralState object on which this function is called must
            have the same parameter shapes as the one who's parameters are being
            loaded.

        :param location: The location to load the NeuralState parameters from.
        :type location: str or file
        """
        state_dict = torch.load(location, map_location=self.device)

        for net in self.networks:
            getattr(self, net).load_state_dict(state_dict[net])

        if hasattr(self, "unitary_dict") and "unitary_dict" in state_dict.keys():
            self.unitary_dict = state_dict["unitary_dict"]

    @staticmethod
    @abc.abstractmethod
    def autoload(location, gpu=False):
        """Initializes a NeuralState from the parameters in the given
        location.

        :param location: The location to load the model parameters from.
        :type location: str or file
        :param gpu: Whether the returned model should be on the GPU.
        :type gpu: bool

        :returns: A new NeuralState initialized from the given parameters.
                  The returned NeuralState will be of whichever type this function
                  was called on. An error may be thrown if the loaded parameters
                  correspond to a different type of NeuralState than the caller.
        """

    @abc.abstractmethod
    def importance_sampling_numerator(self, vp, v):
        r"""Compute the numerator of the weight of sample `vp`,
        with respect to the sample `v`.

        In the case of a mixed state, this quantity is :math:`\rho(\bm{\sigma'}, \bm{\sigma})`,
        while in the pure case it is :math:`\psi(\bm{\sigma'})`

        :param vp: A batch containing the samples :math:`\bm{\sigma'}`
        :type vp: torch.Tensor
        :param v: A batch containing the samples :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: A complex tensor containing the numerator of the weights of
                  :math:`\bm{\sigma'}` with respect to :math:`\bm{\sigma}`
        :rtype: torch.Tensor
        """

    @abc.abstractmethod
    def importance_sampling_denominator(self, v):
        r"""Compute the denominator of the weight of an arbitrary sample,
        with respect to the sample `v`.

        In the case of a mixed state, this quantity is :math:`\rho(\bm{\sigma}, \bm{\sigma})`,
        while in the pure case it is :math:`\psi(\bm{\sigma'})`

        :param v: A batch containing the samples :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: A complex tensor containing the denominator of the weights
                  with respect to :math:`\bm{\sigma}`
        :rtype: torch.Tensor
        """

    def importance_sampling_weight(self, vp, v):
        r"""Compute the weight of sample `vp`, with respect to the sample `v`.

        In the case of a mixed state, this ratio is:

        .. math::
            \frac{\rho(\bm{\sigma'}, \bm{\sigma})}{\rho(\bm{\sigma}, \bm{\sigma})}

        While in the pure case:

        .. math::
            \frac{\psi(\bm{\sigma'})}{\psi(\bm{\sigma})}

        :param vp: A batch containing the samples :math:`\bm{\sigma'}`
        :type vp: torch.Tensor
        :param v: A batch containing the samples :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: A complex tensor containing the weights of :math:`\bm{\sigma'}` with
                  respect to :math:`\bm{\sigma}`
        :rtype: torch.Tensor
        """
        return elementwise_division(
            self.importance_sampling_numerator(vp, v),
            self.importance_sampling_denominator(v),
        )

    def gradient(self, samples, bases=None):
        r"""Compute the gradient of a batch of sample, measured in given bases.

        :param sample: A batch of samples to compute the gradient of.
        :type sample: numpy.ndarray
        :param basis: A batch of bases.
        :type basis: numpy.ndarray or list[str] or None

        :returns: A list of 2 tensors containing the accumulated gradients
                  of each of the internal RBMs.
        :rtype: list[torch.Tensor]
        """
        grad = [
            torch.zeros(
                getattr(self, net).num_pars, dtype=torch.double, device=self.device
            )
            for net in self.networks
        ]
        if bases is None:
            grad[0] = self.rbm_am.effective_energy_gradient(samples)
        else:
            if samples.dim() < 2:
                samples = samples.unsqueeze(0)
                bases = np.array(list(bases)).reshape(1, -1)

            unique_bases, indices = np.unique(bases, axis=0, return_inverse=True)
            indices = torch.Tensor(indices).to(samples)

            for i in range(unique_bases.shape[0]):
                basis = unique_bases[i, :]
                rot_sites = np.where(basis != "Z")[0]

                if rot_sites.size != 0:
                    sample_grad = self.rotated_gradient(basis, samples[indices == i, :])
                else:
                    sample_grad = [
                        self.rbm_am.effective_energy_gradient(samples[indices == i, :]),
                        0.0,
                    ]

                grad[0] += sample_grad[0]  # Accumulate amplitude RBM gradient
                grad[1] += sample_grad[1]  # Accumulate phase RBM gradient

        return grad

    def positive_phase_gradients(self, samples_batch, bases_batch=None):
        r"""Computes the positive phase of the gradients of the parameters.

        :param samples_batch: The measurements
        :type samples_batch: torch.Tensor
        :param bases_batch: The bases in which the measurements are made
        :type bases_batch: numpy.ndarray

        :returns: A two-element list containing the amplitude and phase RBM gradients
        :rtype: list[torch.Tensor]
        """
        grad = self.gradient(samples_batch, bases=bases_batch)
        grad = [gr / float(samples_batch.shape[0]) for gr in grad]
        return grad

    def compute_exact_gradients(self, samples_batch, space, bases_batch=None):
        r"""Computes the gradients of the parameters, using exact sampling
        for the negative phase update instead of Gibbs sampling

        :param samples_batch: The measurements
        :type samples_batch: torch.Tensor
        :param space: A rank 2 tensor of the entire visible space.
        :type space: torch.Tensor
        :param bases_batch: The bases in which the measurements are made
        :type bases_batch: numpy.ndarray

        :returns: A two-element list containing the amplitude and phase RBM
                  gradients calculated with an exact negative phase update
        :rtype: list[torch.Tensor]
        """
        # Positive phase: learning signal driven by the data (and bases)
        grad = self.positive_phase_gradients(samples_batch, bases_batch=bases_batch)

        # Negative phase: learning signal driven by the amplitude RBM of
        # the NN state
        probs = self.probability(space, Z=1.0)  # unnormalized probs
        Z = probs.sum()
        probs /= Z

        all_grads = self.rbm_am.effective_energy_gradient(space, reduce=False)
        grad[0] -= torch.mv(
            all_grads.t(), probs
        )  # average the gradients, weighted by probs

        return grad

    def compute_batch_gradients(self, k, samples_batch, neg_batch, bases_batch=None):
        """Compute the gradients of a batch of the training data (`samples_batch`).

        If measurements are taken in bases other than the reference basis,
        a list of bases (`bases_batch`) must also be provided.

        :param k: Number of contrastive divergence steps in training.
        :type k: int
        :param samples_batch: Batch of the input samples.
        :type samples_batch: torch.Tensor
        :param neg_batch: Batch of the input samples for computing the
                          negative phase.
        :type neg_batch: torch.Tensor
        :param bases_batch: Batch of the input bases corresponding to the samples
                            in `samples_batch`.
        :type bases_batch: numpy.ndarray

        :returns: A two-element list containing the amplitude and phase RBM
                  gradients calculated with a Gibbs sampled negative phase update
        :rtype: list[torch.Tensor]
        """
        # Positive phase: learning signal driven by the data (and bases)
        grad = self.positive_phase_gradients(samples_batch, bases_batch=bases_batch)

        # Negative phase: learning signal driven by the amplitude RBM of
        # the NN state
        vk = self.rbm_am.gibbs_steps(k, neg_batch)
        grad_model = self.rbm_am.effective_energy_gradient(vk)
        grad[0] -= grad_model / float(neg_batch.shape[0])
        # No negative signal for the phase parameters
        return grad

    def _shuffle_data(
            self,
            pos_batch_size,
            neg_batch_size,
            num_batches,
            train_samples,
            input_bases,
            z_samples,
    ):
        pos_batch_perm = torch.randperm(train_samples.shape[0])

        shuffled_pos_samples = train_samples[pos_batch_perm]
        if input_bases is None:
            if neg_batch_size == pos_batch_size:
                neg_batch_perm = pos_batch_perm
            else:
                neg_batch_perm = torch.randint(
                    train_samples.shape[0],
                    size=(num_batches * neg_batch_size,),
                    dtype=torch.long,
                )
            shuffled_neg_samples = train_samples[neg_batch_perm]
        else:
            neg_batch_perm = torch.randint(
                z_samples.shape[0],
                size=(num_batches * neg_batch_size,),
                dtype=torch.long,
            )
            shuffled_neg_samples = z_samples[neg_batch_perm]

        # List of all the batches for positive phase.
        pos_batches = [
            shuffled_pos_samples[batch_start : (batch_start + pos_batch_size)]
            for batch_start in range(0, len(shuffled_pos_samples), pos_batch_size)
        ]

        neg_batches = [
            shuffled_neg_samples[batch_start : (batch_start + neg_batch_size)]
            for batch_start in range(0, len(shuffled_neg_samples), neg_batch_size)
        ]

        if input_bases is not None:
            shuffled_pos_bases = input_bases[pos_batch_perm]
            pos_batches_bases = [
                shuffled_pos_bases[batch_start : (batch_start + pos_batch_size)]
                for batch_start in range(0, len(train_samples), pos_batch_size)
            ]
            return zip(pos_batches, neg_batches, pos_batches_bases)
        else:
            return zip(pos_batches, neg_batches)

    def fit(
            self,
            data,
            epochs=100,
            pos_batch_size=100,
            neg_batch_size=None,
            k=1,
            lr=1e-3,
            input_bases=None,
            progbar=False,
            starting_epoch=1,
            time=False,
            callbacks=None,
            optimizer=torch.optim.SGD,
            optimizer_args=None,
            scheduler=None,
            scheduler_args=None,
            **kwargs,
    ):
        r"""Train the NeuralState.

        :param data: The training samples
        :type data: numpy.ndarray
        :param epochs: The number of full training passes through the dataset.
                       Technically, this specifies the index of the *last* training
                       epoch, which is relevant if `starting_epoch` is being set.
        :type epochs: int
        :param pos_batch_size: The size of batches for the positive phase
                               taken from the data.
        :type pos_batch_size: int
        :param neg_batch_size: The size of batches for the negative phase
                               taken from the data. Defaults to `pos_batch_size`.
        :type neg_batch_size: int
        :param k: The number of contrastive divergence steps.
        :type k: int
        :param lr: Learning rate
        :type lr: float
        :param input_bases: The measurement bases for each sample. Must be provided
                            if training a ComplexWaveFunction or DensityMatrix.
        :type input_bases: numpy.ndarray
        :param progbar: Whether or not to display a progress bar. If "notebook"
                        is passed, will use a Jupyter notebook compatible
                        progress bar.
        :type progbar: bool or str
        :param starting_epoch: The epoch to start from. Useful if continuing training
                               from a previous state.
        :type starting_epoch: int
        :param callbacks: Callbacks to run while training.
        :type callbacks: list[qucumber.callbacks.CallbackBase]
        :param optimizer: The constructor of a torch optimizer.
        :type optimizer: torch.optim.Optimizer
        :param scheduler: The constructor of a torch scheduler
        :param optimizer_args: Arguments to pass to the optimizer
        :type optimizer_args: dict
        :param scheduler_args: Arguments to pass to the scheduler
        :type scheduler_args: dict
        :param \**kwargs: Ignored; exists for backwards compatibility.
        """
        if self.stop_training:  # terminate immediately if stop_training is true
            return

        disable_progbar = progbar is False
        progress_bar = tqdm_notebook if progbar == "notebook" else tqdm

        callbacks = CallbackList(callbacks if callbacks else [])
        if time:
            callbacks.append(Timer())

        neg_batch_size = neg_batch_size if neg_batch_size else pos_batch_size

        if isinstance(data, torch.Tensor):
            train_samples = (
                data.clone().detach().to(device=self.device, dtype=torch.double)
            )
        else:
            train_samples = torch.tensor(data, device=self.device, dtype=torch.double)

        all_params = [getattr(self, net).parameters() for net in self.networks]
        all_params = list(chain(*all_params))

        optimizer_args = {} if optimizer_args is None else optimizer_args
        scheduler_args = {} if scheduler_args is None else scheduler_args

        optimizer = optimizer(all_params, lr=lr, **optimizer_args)

        if scheduler is not None:
            scheduler = scheduler(optimizer, **scheduler_args)

        if input_bases is not None:
            z_samples = extract_refbasis_samples(train_samples, input_bases).to(
                device=self.device
            )
        else:
            z_samples = None

        callbacks.on_train_start(self)

        num_batches = ceil(train_samples.shape[0] / pos_batch_size)
        for ep in progress_bar(
                range(starting_epoch, epochs + 1), desc="Epochs ", disable=disable_progbar
        ):
            data_iterator = self._shuffle_data(
                pos_batch_size,
                neg_batch_size,
                num_batches,
                train_samples,
                input_bases,
                z_samples,
            )
            callbacks.on_epoch_start(self, ep)

            for b, batch in enumerate(data_iterator):
                callbacks.on_batch_start(self, ep, b)

                all_grads = self.compute_batch_gradients(k, *batch)

                optimizer.zero_grad()  # clear any cached gradients

                # assign gradients to corresponding parameters
                for i, net in enumerate(self.networks):
                    rbm = getattr(self, net)
                    vector_to_grads(all_grads[i], rbm.parameters())

                optimizer.step()  # tell the optimizer to apply the gradients

                callbacks.on_batch_end(self, ep, b)
                if self.stop_training:  # check for stop_training signal
                    break

            if scheduler is not None:
                scheduler.step()

            callbacks.on_epoch_end(self, ep)
            if self.stop_training:  # check for stop_training signal
                break

        callbacks.on_train_end(self)


# make module path show up properly in sphinx docs
NeuralStateBase.__module__ = "qucumber.nn_states"


class WaveFunctionBase(NeuralStateBase):
    """Abstract Base Class for WaveFunctions."""

    def reinitialize_parameters(self):
        """Randomize the parameters of the internal RBMs."""
        for net in self.networks:
            getattr(self, net).initialize_parameters()

    def __getattr__(self, attr):
        return getattr(self.rbm_am, attr)

    def amplitude(self, v):
        r"""Compute the (unnormalized) amplitude of a given vector/matrix of visible states.

        .. math::

            \text{amplitude}(\bm{\sigma})=|\psi(\bm{\sigma})|

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Matrix/vector containing the amplitudes of v
        :rtype: torch.Tensor
        """
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    @abc.abstractmethod
    def phase(self, v):
        r"""Compute the phase of a given vector/matrix of visible states.

        .. math::
            \text{phase}(\bm{\sigma})

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Matrix/vector containing the phases of v
        :rtype: torch.Tensor
        """

    def psi(self, v):
        r"""Compute the (unnormalized) wavefunction of a given vector/matrix of
        visible states.

        .. math::
            \psi(\bm{\sigma})

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Complex object containing the value of the wavefunction for
                  each visible state
        :rtype: torch.Tensor
        """
        # vectors/tensors of shape (len(v),)
        amplitude, phase = self.amplitude(v), self.phase(v)
        return make_complex(
            amplitude * phase.cos(),  # real part
            amplitude * phase.sin(),  # imaginary part
        )

    def importance_sampling_numerator(self, vp, v):
        return self.psi(vp)

    def importance_sampling_denominator(self, v):
        return self.psi(v)


# make module path show up properly in sphinx docs
WaveFunctionBase.__module__ = "qucumber.nn_states"


class ComplexWaveFunction(WaveFunctionBase):
    """Class capable of learning wavefunctions with a non-zero phase.

    :param num_visible: The number of visible units, ie. the size of the system being learned.
    :type num_visible: int
    :param num_hidden: The number of hidden units in both internal RBMs. Defaults to
                    the number of visible units.
    :type num_hidden: int
    :param unitary_dict: A dictionary mapping unitary names to their matrix representations.
    :type unitary_dict: dict[str, torch.Tensor]
    :param gpu: Whether to perform computations on the default GPU.
    :type gpu: bool
    :param module: An instance of a BinaryRBM module to use for density estimation;
                   The given RBM object will be used to estimate the amplitude of
                   the wavefunction, while a copy will be used to estimate
                   the phase of the wavefunction.
                   Will be copied to the default GPU if `gpu=True` (if it
                   isn't already there). If `None`, will initialize the BinaryRBMs
                   from scratch.
    :type module: qucumber.rbm.BinaryRBM
    """

    _rbm_am = None
    _rbm_ph = None
    _device = None

    def __init__(
            self, num_visible, num_hidden=None, unitary_dict=None, gpu=False, module=None
    ):
        if gpu and torch.cuda.is_available():
            warnings.warn(
                "Using ComplexWaveFunction on GPU is not recommended due to poor performance compared to CPU.",
                ResourceWarning,
                2,
            )
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if module is None:
            self.rbm_am = BinaryRBM(num_visible, num_hidden, gpu=gpu)
            self.rbm_ph = BinaryRBM(num_visible, num_hidden, gpu=gpu)
        else:
            self.rbm_am = module.to(self.device)
            self.rbm_am.device = self.device
            self.rbm_ph = module.to(self.device).clone()
            self.rbm_ph.device = self.device

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden
        self.device = self.rbm_am.device

        self.unitary_dict = unitary_dict if unitary_dict else create_dict()
        self.unitary_dict = {
            k: v.to(device=self.device) for k, v in self.unitary_dict.items()
        }

    @property
    def networks(self):
        return ["rbm_am", "rbm_ph"]

    @property
    def rbm_am(self):
        return self._rbm_am

    @rbm_am.setter
    def rbm_am(self, new_val):
        self._rbm_am = new_val

    @property
    def rbm_ph(self):
        """RBM used to learn the wavefunction phase."""
        return self._rbm_ph

    @rbm_ph.setter
    def rbm_ph(self, new_val):
        self._rbm_ph = new_val

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_val):
        self._device = new_val

    def amplitude(self, v):
        r"""Compute the (unnormalized) amplitude of a given vector/matrix of visible states.

        .. math::

            \text{amplitude}(\bm{\sigma})=|\psi_{\bm{\lambda\mu}}(\bm{\sigma})|=
            e^{-\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})/2}

        :param v: visible states :math:`\bm{\sigma}`.
        :type v: torch.Tensor

        :returns: Vector containing the amplitudes of the given states.
        :rtype: torch.Tensor
        """
        return super().amplitude(v)

    def phase(self, v):
        r"""Compute the phase of a given vector/matrix of visible states.

        .. math::

            \text{phase}(\bm{\sigma})=-\mathcal{E}_{\bm{\mu}}(\bm{\sigma})/2

        :param v: visible states :math:`\bm{\sigma}`.
        :type v: torch.Tensor

        :returns: Vector containing the phases of the given states.
        :rtype: torch.Tensor
        """
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi(self, v):
        r"""Compute the (unnormalized) wavefunction of a given vector/matrix of visible states.

        .. math::

            \psi_{\bm{\lambda\mu}}(\bm{\sigma})
                = e^{-[\mathcal{E}_{\bm{\lambda}}(\bm{\sigma})
                        + i\mathcal{E}_{\bm{\mu}}(\bm{\sigma})]/2}

        :param v: visible states :math:`\bm{\sigma}`
        :type v: torch.Tensor

        :returns: Complex object containing the value of the wavefunction for
                  each visible state
        :rtype: torch.Tensor
        """
        return super().psi(v)

    def rotated_gradient(self, basis, sample):
        r"""Computes the gradients rotated into the measurement basis

        :param basis: The bases in which the measurement is made
        :type basis: numpy.ndarray
        :param sample: The measurement (either 0 or 1)
        :type sample: torch.Tensor

        :returns: A list of two tensors, representing the rotated gradients
                  of the amplitude and phase RBMS
        :rtype: list[torch.Tensor, torch.Tensor]
        """
        Upsi, Upsi_v, v = rotate_psi_inner_prod(
            self, basis, sample, include_extras=True
        )
        inv_Upsi = inverse(Upsi)

        raw_grads = [self.am_grads(v), self.ph_grads(v)]

        rotated_grad = [einsum("ib,ibg->bg", Upsi_v, g) for g in raw_grads]
        grad = [
            einsum("b,bg->g", inv_Upsi, rg, imag_part=False) for rg in rotated_grad
        ]

        return grad

    def am_grads(self, v):
        r"""Computes the gradients of the amplitude RBM for given input states

        :param v: The input state, :math:`\sigma`
        :type v: torch.Tensor

        :returns: The gradients of all amplitude RBM parameters
        :rtype: torch.Tensor
        """
        return make_complex(self.rbm_am.effective_energy_gradient(v, reduce=False))

    def ph_grads(self, v):
        r"""Computes the gradients of the phase RBM for given input states

        :param v: The input state, :math:`\sigma`
        :type v: torch.Tensor

        :returns: The gradients of all phase RBM parameters
        :rtype: torch.Tensor
        """
        return scalar_mult(
            make_complex(self.rbm_ph.effective_energy_gradient(v, reduce=False)),
            I,  # need to multiply phase gradient by i
        )

    def fit(
            self,
            data,
            epochs=100,
            pos_batch_size=100,
            neg_batch_size=None,
            k=1,
            lr=1e-3,
            input_bases=None,
            progbar=False,
            starting_epoch=1,
            time=False,
            callbacks=None,
            optimizer=torch.optim.SGD,
            optimizer_args=None,
            scheduler=None,
            scheduler_args=None,
            **kwargs
    ):
        if input_bases is None:
            raise ValueError(
                "input_bases must be provided to train a ComplexWaveFunction!"
            )
        else:
            super().fit(
                data=data,
                epochs=epochs,
                pos_batch_size=pos_batch_size,
                neg_batch_size=neg_batch_size,
                k=k,
                lr=lr,
                input_bases=input_bases,
                progbar=progbar,
                starting_epoch=starting_epoch,
                time=time,
                callbacks=callbacks,
                optimizer=optimizer,
                optimizer_args=optimizer_args,
                scheduler=scheduler,
                scheduler_args=scheduler_args,
                **kwargs
            )
