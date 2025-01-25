#### Warm Up | Master's Thesis

# MULTI-QUBIT BLOCH REPRESENTATION



```
DISCLAIMER!

After an initial proof sketch, large part of the derivation was done by GPT-o1 and then manually edited.
```



#### DERIVATION SKETCH

The **single-qubit Bloch representation** is typically mentioned at the beginning of literature. We want to get to the **multi-qubit expansion** as used in the notebook:
$$
\rho_{123\ldots N}
=\frac{1}{2^N}\sum_{i_1, i_2, \ldots, i_N=0}^3
\Bigl\langle \sigma_{i_1} \otimes \sigma_{i_2} \otimes \cdots \otimes \sigma_{i_N}\Bigr\rangle 
\;\Bigl(\sigma_{i_1} \otimes \sigma_{i_2} \otimes \cdots \otimes \sigma_{i_N}\Bigr).
$$

The key ideas are:

1. Recognizing that $\sigma_0,\sigma_1,\sigma_2,\sigma_3$ forms an orthonormal operator basis for the space of $2\times 2$ matrices.
2. Generalizing to $N$ qubits by taking all tensor products of these single-qubit basis operators (yielding $4^N$ basis elements).
3. Ensuring normalization by including the factor $1/2^N$.

---



#### SINGLE QUBIT BLOCH REPRESENTATION

**Operator Basis for 1 Qubit**

For one qubit, every $2\times 2$ matrix can be expanded in the basis
$$
\{\sigma_0, \sigma_1, \sigma_2, \sigma_3\},
$$

These satisfy the orthonormality condition under the Hilbert–Schmidt inner product
$$
\left\langle\sigma_i, \sigma_j\right\rangle_{\mathrm{HS}}=\text{tr}(\sigma_i^\dagger \sigma_j) = 2\ \delta_{ij}
$$
Because $\sigma_i$ are Hermitian and traceless (except $\sigma_0=I$), they form a convenient basis.



**Expanding a Single-Qubit Density Matrix**

Any single-qubit density matrix $\rho$ (which is a positive semi-definite, Hermitian $2\times 2$ operator with unit trace) can be written as a linear combination of these basis operators. We write
$$
\rho \;=\; \sum_{\alpha=0}^{3} c_\alpha\,\sigma_\alpha.
$$

We want the expansion to reproduce $\rho$ correctly and ensure $\operatorname{tr}(\rho)=1$. Using the orthonormality of $\sigma_\alpha$, one finds

$$
\left\langle\rho, \sigma_\alpha\right\rangle_{\mathrm{HS}}=\left\langle\sum_{\beta=0}^3 c_\beta \sigma_\beta, \sigma_\alpha\right\rangle_{\mathrm{HS}},
$$

which expands to

$$
\left\langle\rho, \sigma_\alpha\right\rangle_{\mathrm{HS}}=\sum_{\beta=0}^3 c_\beta\left\langle\sigma_\beta, \sigma_\alpha\right\rangle_{\mathrm{HS}} .
$$


Using the orthonormality relation $\left\langle\sigma_\beta, \sigma_\alpha\right\rangle_{\mathrm{HS}}=2 \delta_{\alpha \beta}$, we remain only with the on sum term where $\alpha = \beta$

$$
\left\langle\rho, \sigma_\alpha\right\rangle_{\mathrm{HS}}=\operatorname{tr}\left(\rho \sigma_\alpha\right)= 2c_\alpha,
$$


So

$$
c_\alpha=\frac{1}{2} \operatorname{tr}\left(\rho \sigma_\alpha\right) .
$$
Hence,
$$
\rho 
= \sum_{\alpha=0}^{3} \left[ \frac{1}{2}\,\text{tr}(\rho\,\sigma_\alpha) \right] \sigma_\alpha
=\frac{1}{2}\sum_{\alpha=0}^{3} \text{tr}(\rho\,\sigma_\alpha)\,\sigma_\alpha.
$$



**Usual Bloch Form**

Often, one separates the coefficient of the identity $\sigma_0=I$ and groups the rest into a “Bloch vector” $\vec{s}=(s_1,s_2,s_3)$. Because
$$
s_\alpha=\operatorname{tr}\left(\rho \sigma_\alpha\right) \quad \quad \text{for} \quad \alpha=1,2,3
$$
we get
$$
\rho 
= \frac{1}{2}\Bigl(\sigma_0 + \sum_{\alpha=1}^{3} s_\alpha\,\sigma_\alpha\Bigr)
= \frac{1}{2}\Bigl(I + \vec{s}\cdot\vec{\sigma}\Bigr).
$$

That is precisely the single-qubit Bloch representation w esee so often in the literature.

---



#### FROM SINGLE QUBIT TO N QUBITS

**Tensor Products as a Basis for $N$-Qubit Operators**

For **$N$ qubits**, the overall Hilbert space has dimension $2^N$. Consequently, operators acting on this space are $2^N \times 2^N$ matrices. A natural basis for these matrices arises from **all possible tensor products** of the single-qubit basis operators:

$$
\bigl\{
\sigma_{i_1} \otimes \sigma_{i_2} \otimes \cdots \otimes \sigma_{i_N}
\;\big|\;
i_m \in \{0,1,2,3\}\bigr\}.
$$

There are $4^N$ such tensor-product operators, each of size $2^N \times 2^N$. Just as for the single-qubit case, one can show that these operators form an orthonormal basis under the Hilbert–Schmidt inner product:

$$
\text{tr}\bigl(
(\sigma_{i_1}\otimes\cdots\otimes\sigma_{i_N})^\dagger
(\sigma_{j_1}\otimes\cdots\otimes\sigma_{j_N})
\bigr)
= 2^N\,\delta_{i_1j_1}\cdots\delta_{i_N j_N}
$$

Here we see the factor $2^N$ which needs to be counter acted by $1/2^N$ to achieve proper unit trace.



**Expanding an $N$-Qubit Density Matrix**

Given that $\rho_{12\ldots N}$ is an $N$-qubit density operator (i.e., a $2^N \times 2^N$ positive semi-definite, Hermitian operator with trace 1), we expand it in this basis:

$$
\rho_{12\ldots N}
= \sum_{i_1=0}^3 \sum_{i_2=0}^3 \cdots \sum_{i_N=0}^3
\; c_{\,i_1,i_2,\ldots,i_N} 
\bigl(\sigma_{i_1}\otimes \sigma_{i_2}\otimes\cdots\otimes \sigma_{i_N}\bigr).
$$

Using the same logic as in the single-qubit case, the coefficient in front of each tensor product is found by taking the trace:

$$
c_{\,i_1,i_2,\ldots,i_N}
\;=\;
\frac{1}{2^N}\,\text{tr}
\Bigl(
\rho_{12\ldots N}\,\bigl[\sigma_{i_1}\otimes\cdots\otimes\sigma_{i_N}\bigr]
\Bigr).
$$

Now we already get a term very close to the desired form:
$$
\rho_{12\ldots N}
\;=\;
\frac{1}{2^N}\sum_{i_1=0}^3 \sum_{i_2=0}^3 \cdots \sum_{i_N=0}^3
\text{tr}\Bigl(\rho_{12\ldots N}\,\sigma_{i_1}\otimes\cdots\otimes\sigma_{i_N}\Bigr)
\;\sigma_{i_1}\otimes\cdots\otimes\sigma_{i_N}.
$$

A very similar formula can be found in `Nielsen et al. Eq. 8.149, p. 390`.



**Identifying the Expectation Values**

We often write
$$
\langle \sigma_{i_1}\otimes\cdots\otimes\sigma_{i_N}\rangle
\;\;=\;\;
\text{tr}\Bigl(\rho_{12\ldots N}\,\sigma_{i_1}\otimes\cdots\otimes\sigma_{i_N}\Bigr),
$$
calling these the “expectation values” of the measurement operators $\sigma_{i_1}\otimes\cdots\otimes\sigma_{i_N}$. Substituting back gives:
$$
\rho_{12\ldots N}
= \frac{1}{2^N} 
\sum_{i_1=0}^3 \sum_{i_2=0}^3 \cdots \sum_{i_N=0}^3
\Bigl\langle\sigma_{i_1}\otimes\cdots\otimes\sigma_{i_N}\Bigr\rangle
\;\Bigl(\sigma_{i_1}\otimes\cdots\otimes\sigma_{i_N}\Bigr).
$$

**3 QUBIT CASE**

From this general formula we are now able to construct the formula in the notebook:
$$
\rho_{123}
= \frac{1}{2^3} \sum_{i,j,k=0}^{3} 
\Bigl\langle 
\sigma_{i}\otimes \sigma_{j}\otimes \sigma_{k}
\Bigr\rangle
\,\Bigl(
\sigma_{i}\otimes \sigma_{j}\otimes \sigma_{k}
\Bigr).
$$
This follows immediately by letting $N=3$ and relabeling the indices as $(i_1,i_2,i_3)=(i,j,k)$.

---



#### GROUPING BY CORRELATION ORDER

Many textbooks adopt a more conceptual or physically motivated grouping (eg. `Hassan et al., Eq. 20`). They group the terms according to which qubits carry a Pauli operator (rather than the identity):

$$
\rho_{12 \ldots N} =\frac{1}{2^N}\Bigl[
I + \sum_{k=1}^N s^{(k)} \sigma^{(k)} + \sum_{\{k_1, k_2\}} T^{\{k_1, k_2\}} \sigma^{(k_1)}\sigma^{(k_2)} + \ldots + T^{\{1,2,\ldots,N\}} \sigma^{(1)}\cdots\sigma^{(N)} \Bigr]
$$

- The *first term* $I$ is the identity on *all* $N$ qubits (which, in the other notation, corresponds to $i_1 = \ldots = i_N = 0$).  
- The *second term* $\sum_{k=1}^N s^{(k)}\sigma^{(k)}$ picks out those terms that act non-trivially on **exactly one** qubit $k$ (i.e., all other qubits have the identity operator).  
- The *third term* $\sum_{\{k_1, k_2\}} T^{\{k_1, k_2\}}\sigma^{(k_1)}\sigma^{(k_2)}$ groups *all* those terms that act non-trivially on exactly two qubits $k_1$ and $k_2$, while the remaining qubits are identity. 
- And so forth, up through the term $T^{\{1,2,\ldots,N\}}\sigma^{(1)}\cdots\sigma^{(N)}$, which is the single term that acts non-trivially (Pauli) on *all* $N$ qubits.



This “hierarchical” version:
1. **Directly shows** how each term corresponds to 1-qubit observables, 2-qubit correlations, etc.  
2. **Highlights** local vs. nonlocal (correlation) parts and ties in with discussions of entanglement.



In **more detail**, each $T^{\{k_1, \dots, k_m\}}$ is just the corresponding **expectation value** in the standard Bloch picture, but explicitly grouped by how many qubits have non-identity operators.  

Specifically,
$$
T^{\{k_1,\ldots,k_m\}}
\;=\;
\text{tr}\Bigl(
\rho_{12\ldots N}\,
\sigma_0 \otimes \cdots \otimes \sigma_0
\otimes \underset{k_1}{\sigma_{i_{k_1}}}
\otimes \cdots
\otimes \underset{k_m}{\sigma_{i_{k_m}}}
\otimes \cdots
\otimes \sigma_0
\Bigr).
$$

Here, each qubit **not** in $\{k_1,\dots,k_m\}$ is assigned $\sigma_0 = I$ (the identity), while for qubits $k_1,\dots,k_m$ we insert one of $\{\sigma_x, \sigma_y, \sigma_z\}$. Thus, the “\(T\)” notation simply **repackages** the same trace coefficients in a way that highlights the local (single-qubit) and multi-qubit (correlation) parts.



#### REFERENCES

- Nielsen, M. A., & Chuang, I. L. (2000) *Quantum Computation and Quantum Information*. Cambridge University Press.
- Hassan, A. S. M., & Joag, P. S. (2012) *Geometric measure of quantum discord and total quantum correlations in a N-partite quantum state*. Department of Physics, University of Pune, India.







