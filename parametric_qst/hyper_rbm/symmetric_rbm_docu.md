#### Documentation | Symmetric RBM
# MATHEMATICAL DERIVATION AND IMPLEMENTATION


> *REMARK*: For simplicity reasons, we dropped the temperature $T$ in the derivations.

#### TRAINING OBJECTIVE: SYMMETRIZED FREE ENERGY

To enforce spin-inversion symmetry in the model, we adopt a symmetrized ansatz. We define the probability of a visible configuration $\mathbf{v}$ as a mixture of the configuration itself and its flipped counterpart $1-\mathbf{v}$. The marginal probability is defined as:

$$
p_{\text{sym}}(\mathbf{v}) \propto \exp(-F(\mathbf{v})) + \exp(-F(1-\mathbf{v}))
$$

where $F(\mathbf{v})$ is the standard RBM Free Energy.

**Efficient Computation Strategy**

The standard Free Energy $F(\mathbf{v})$ requires computing the matrix-vector product $\mathbf{v}^{\top} \mathbf{W}$ and for the flipped state we would need to compute $(1-\mathbf{v})^{\top} \mathbf{W}$. We utilize the linearity of the matrix product to compute the flipped term by reusing the original product:

$$
(1-\mathbf{v})^{\top} \mathbf{W} = \mathbf{1}^{\top} \mathbf{W} - \mathbf{v}^{\top} \mathbf{W} = \mathbf{\Sigma}_W^{\top} - \mathbf{v}^{\top} \mathbf{W}
$$

where $\mathbf{\Sigma}_W$ is the vector of column sums of $\mathbf{W}$ (implemented as `W_colsum`).

The final symmetrized energy sums over two exponentials inside a logarithm:

$$
F_{\text{sym}}(\mathbf{v}) = -\ln \left[ \exp\left(-F(\mathbf{v})\right) + \exp\left(-F(1-\mathbf{v})\right) \right]
$$

We use the LogSumExp trick to maintain numerical stability when computing this expression:

$$
F_{\text{sym}}(\mathbf{v}) = - \text{logsumexp}\left(\begin{bmatrix} -F(\mathbf{v}) \\ -F(1-\mathbf{v}) \end{bmatrix} \right)
$$


#### GIBBS SAMPLING WITH A LATENT SYMMETRY VARIABLE

**Broken Independence Problem**

While $F_{\text{sym}}(\mathbf{v})$ allows us to compute gradients for training, it is unsuitable for direct Gibbs sampling because the symmetrization breaks the conditional independence of the hidden units.

Unlike a standard RBM where the conditional probability factorizes as  $p(\mathbf{h} \mid \mathbf{v}) = \prod_j p(h_j \mid \mathbf{v})$, the symmetrized distribution becomes a sum of products:

$$
p(\mathbf{h} \mid \mathbf{v}) \propto \underbrace{\left(\prod_j p(h_j \mid \mathbf{v})\right)}_{\text {canonical branch}}+\underbrace{\left(\prod_j p(h_j \mid 1-\mathbf{v})\right)}_{\text {flipped branch}}
$$

Since $\sum \prod \neq \prod \sum$, the hidden units become dependent on each other, preventing efficient parallel sampling.

**Data Augmentation with a Symmetry Variable**

To resolve this, we introduce a latent "symmetry variable" $u \in \{0, 1\}$. We define an augmented joint distribution over the triplet $(\mathbf{v}, \mathbf{h}, u)$:

$$
p(\mathbf{v}, \mathbf{h}, u) \propto \begin{cases} \text{exp}\left[-E(\mathbf{v}, \mathbf{h})\right] & \text{if } u=1 \text{ (canonical)} \\ \text{exp}\left[-E(1-\mathbf{v}, \mathbf{h})\right] & \text{if } u=0 \text{ (flipped)} \end{cases}
$$

Marginalizing out $u$ recovers the exact symmetrized target distribution. Crucially, conditioning on $u$ restores the bipartite structure of the RBM graph.

**Three-Step Gibbs Sampling**

The Gibbs sampling procedure in `_gibbs_step` follows a specific three-stage update order ($u_{\text{old}} \to \mathbf{h} \to u_{\text{new}} \to \mathbf{v}_{\text{new}}$) to ensure proper mixing.

> *Step 1: Sample Hidden Units $(\mathbf{v}, u \to \mathbf{h})$*:
>
> If we flipped the input $\mathbf{v}$ in the previous Gibbs step, we need to align it back to the canonical orientation before sampling $\mathbf{h}$:
> $$\mathbf{v}_{\text {canon}}= \begin{cases}\mathbf{v} & \text { if } u=1 \\ 1-\mathbf{v} & \text { if } u=0\end{cases}$$
> This is necessary because the weights $\mathbf{W}$ are learned in a fixed canonical orientation. The hidden units are then sampled normally given this aligned input:
> $$p\left(h_j=1 \mid \mathbf{v}, u\right)=p\left(h_j=1 \mid \mathbf{v}_\text{canon}\right)=\text{sigmoid}\left(\left(\mathbf{v}_{\text {canon }}^{\top} \mathbf{W}\right)_j+c_j\right)$$

> *Step 2: Update Symmetry Variable $(\mathbf{v}, \mathbf{h} \to u)$*:
>
> With the newly sampled features $\mathbf{h}$, we must decide which symmetry branch best explains the joint state $(\mathbf{v}, \mathbf{h})$. We are interested in the log-odds favoring the canonical branch. The probability is basically a sigmoid over the energy difference $\Delta E$ between the two branches:
> $$
\begin{aligned}
p(u=1 \mid \mathbf{v}, \mathbf{h})
&= \frac{\exp\!\left(-E(\mathbf{v},\mathbf{h})\right)}{\exp\!\left(-E(\mathbf{v},\mathbf{h})\right)+\exp\!\left(-E(1-\mathbf{v},\mathbf{h})\right)} \\
&= \text{sigmoid}\!\left(E(1-\mathbf{v},\mathbf{h}) - E(\mathbf{v},\mathbf{h})\right) \\
&= \text{sigmoid}(\Delta E)
\end{aligned}
$$
> This corresponds to the log-odds favoring the canonical branch.


**Intermezzo: Calculating $\Delta E$ with reusable Matrix Product**

We use the $0/1$ visible convention and the vanilla RBM joint energy:

$$ E(\mathbf{v},\mathbf{h}) = -\mathbf{v}^T\mathbf{b} - \mathbf{h}^T\mathbf{c} - \mathbf{v}^T \mathbf{W}\mathbf{h} $$

We define the pre-activation field $\mathbf{a} := \mathbf{W}\mathbf{h}$. Substituting this simplifies the energy to:

$$ E(\mathbf{v},\mathbf{h}) = -\mathbf{v}^T\mathbf{b} - \mathbf{h}^T\mathbf{c} - \mathbf{v}^T\mathbf{a} $$

We need $\Delta E = E(1-\mathbf{v},\mathbf{h}) - E(\mathbf{v},\mathbf{h})$. Expanding the energy of the flipped state:

$$
\begin{aligned}
E(1-\mathbf{v},\mathbf{h}) &= -(1-\mathbf{v})^T\mathbf{b} - \mathbf{h}^T\mathbf{c} - (1-\mathbf{v})^T\mathbf{a} \\
&= -\mathbf{1}^T\mathbf{b} + \mathbf{v}^T\mathbf{b} - \mathbf{h}^T\mathbf{c} - \mathbf{1}^T\mathbf{a} + \mathbf{v}^T\mathbf{a}
\end{aligned}
$$

Subtracting the original energy $E(\mathbf{v}, \mathbf{h})$ cancels the hidden bias term $-\mathbf{h}^T\mathbf{c}$. Collecting the remaining terms yields:
$$ \Delta E = -\mathbf{1}^T\mathbf{b} - \mathbf{1}^T\mathbf{a} + 2\mathbf{v}^T\mathbf{b} + 2\mathbf{v}^T\mathbf{a} $$
This is expressed in the code as `dE = -bsum - asum + 2*vb + 2*va`.

The big advantage here is, that we can reuse the pre-computed field $\mathbf{a} = \mathbf{W}\mathbf{h}$ in the subsequent visible sampling step.


> *Step 3: Sample Visible Units $(\mathbf{h}, u \to \mathbf{v})$*:
>
> Finally, we sample the new visible layer. First we sample the canonical visible units elementwise:
> $$  
\begin{aligned}  
p\left((v_{\text{canon}})_i = 1 \mid \mathbf{h}\right)  
&= \text{sigmoid}\left((\mathbf{W}\mathbf{h})_i + b_i\right) \\  
&= \text{sigmoid}\left(a_i + b_i\right)  
\end{aligned}  
$$
> We then apply the symmetry mapping based on $u$:
>
> $$\mathbf{v} \leftarrow  
\begin{cases}  
\mathbf{v}_{\text{canon}} & \text{if } u=1 \\  
1 - \mathbf{v}_{\text{canon}} & \text{if } u=0  
\end{cases}  
$$
> This induces the final conditional distribution $p(v_i \mid \mathbf{h}, u)$. The RBM returns a sample from the proper symmetry sector based on the current state of the chain.


**Initialization of the Symmetry Variable**

During chain initialization we either start from data visibles $\mathbf{v}$ (training/CD) or from random noise (generation). The sampler then needs a symmetry choice to orient the visible state before the first hidden update. We therefore initialize $u$ from an uninformative prior, independently per chain:

$$  
p(u=1)=p(u=0)=0.5.  
$$

After this, Step 1 samples $\mathbf{h}$ given $(\mathbf{v},u)$, so $\mathbf{h}$ does not require a meaningful initialization. This random initialization allows the chain to break symmetry spontaneously during the first few steps as it begins to form structured hidden representations.