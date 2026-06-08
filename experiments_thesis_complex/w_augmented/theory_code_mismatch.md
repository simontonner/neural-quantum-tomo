#### Theory note | Complex RBM tomography
# IMPLEMENTATION-THEORY ALIGNMENT FOR THE PCD CODE

#### Scope

This note concerns the **training objective only**. The key question is whether the implemented loss is a faithful stochastic surrogate of the practical objective introduced in Chapter 6.

The answer is **yes**, provided that the measurement datasets have **equal shot counts per basis**.

#### The theoretical practical objective

The practical training objective in Chapter 6 is

$$
\mathcal{L}_{\lambda,\mu}
=
\mathcal{L}^{\mathrm{data}}_{\lambda} +
\mathcal{L}^{\mathrm{model}}_{\lambda} +
\mathcal{L}^{\mathrm{data,rot}}_{\lambda,\mu},
$$

with

$$
\mathcal{L}^{\mathrm{data}}_{\lambda}
=
\frac{1}{|\mathcal{D}_0|}
\sum_{\mathbf{s}^{[0]}\in\mathcal{D}_0}
F_\lambda(\mathbf{s}^{[0]}),
$$

$$
\mathcal{L}^{\mathrm{data,rot}}_{\lambda,\mu}
= -
\sum_{r=1}^{R}
\frac{1}{|\mathcal{D}_r|}
\sum_{\mathbf{s}^{[r]}\in\mathcal{D}_r}
\ln
\left|
\sum_{\mathbf{s}}
U_r(\mathbf{s}^{[r]},\mathbf{s})
\exp\left(-\frac{1}{2}F_\lambda(\mathbf{s})\right)
\exp\left(-\frac{i}{2}F_\mu(\mathbf{s})\right)
\right|^2,
$$

and

$$
\mathcal{L}^{\mathrm{model}}_{\lambda}
= -
\frac{R+1}{|\tilde{\mathcal{D}}_0|}
\sum_{\tilde{\mathbf{s}}^{[0]}\in\tilde{\mathcal{D}}_0}
F_\lambda(\tilde{\mathbf{s}}^{[0]}).
$$

Thus, the exact multi-basis data likelihood is preserved, while the intractable amplitude-normalization contribution is replaced by a CD-style surrogate.

#### How the code forms minibatches

The loader stacks all measurement files into one pooled dataset,

$$
\texttt{values}
=
\begin{bmatrix}
\text{basis 0 samples} \\
\text{basis 1 samples} \\
\vdots \\
\text{basis R samples}
\end{bmatrix},
$$

and stores the corresponding basis labels alongside them. Training minibatches are then obtained by globally shuffling this pooled sample list and slicing out batches of size $B$.

So operationally, one minibatch is a **mixed positive batch**. However, inside the loss, the samples are grouped again by basis and treated correctly basis-by-basis:

- computational-basis samples contribute $F_\lambda(\mathbf{s})$
- rotated-basis samples contribute $-\log |A_r(\mathbf{s}^{[r]})|^2$

where

$$
A_r(\mathbf{s}^{[r]})
=
\sum_{\mathbf{s}}
U_r(\mathbf{s}^{[r]},\mathbf{s})
\exp\left(-\frac{1}{2}F_\lambda(\mathbf{s})\right)
\exp\left(-\frac{i}{2}F_\mu(\mathbf{s})\right).
$$

The implemented positive term is therefore exact at the sample level. The only stochasticity comes from minibatching.

#### Where the factor $(R+1)$ goes

At first sight, the theory seems to weight only the model term by $(R+1)$, while the implementation uses

$$
\texttt{loss} = \texttt{pos\_loss} - \texttt{neg\_loss}
$$

with both terms formed as means. This can look like a relative-scale mismatch, but in the equal-shot setting it is not.

The reason is that the theoretical positive part is a **sum of per-basis averages**, while the implementation uses a **single pooled average** over all samples.

If all basis datasets have the same number of shots, say

$$
|\mathcal{D}_0| = |\mathcal{D}_1| = \cdots = |\mathcal{D}_R| = N,
$$

then the pooled positive average becomes

$$
\frac{1}{(R+1)N}
\sum_{r=0}^{R}
\sum_{\mathbf{s}^{[r]}\in\mathcal{D}_r}
\ell_r(\mathbf{s}^{[r]})
=
\frac{1}{R+1}
\sum_{r=0}^{R}
\frac{1}{|\mathcal{D}_r|}
\sum_{\mathbf{s}^{[r]}\in\mathcal{D}_r}
\ell_r(\mathbf{s}^{[r]}),
$$

where $\ell_r$ denotes the corresponding per-sample contribution.

So the **positive side is implicitly divided by $(R+1)$** by pooling.

At the same time, the theoretical model term contains an explicit factor $(R+1)$, while the implementation uses only the plain model average. Hence the implemented model term is also smaller by the same factor $\frac{1}{R+1}$.

Therefore, with equal shots per basis, the implemented loss is simply

$$
\mathcal{L}_{\mathrm{impl}}
=
\frac{1}{R+1}\,
\mathcal{L}_{\mathrm{theory}}.
$$

This is a **global constant rescaling**, not a relative reweighting of different parts of the objective.

#### Why this does not matter

A global constant factor does not change the optimum of the objective. It only rescales the gradients:

$$
\nabla \mathcal{L}_{\mathrm{impl}}
=
\frac{1}{R+1}\,
\nabla \mathcal{L}_{\mathrm{theory}}.
$$

For plain SGD, this can be compensated by the learning rate. Thus, in the equal-shot setting, the pooled implementation remains theoretically aligned with the practical objective.

#### Why equal shot counts matter

The above argument relies crucially on

$$
|\mathcal{D}_0| = |\mathcal{D}_1| = \cdots = |\mathcal{D}_R|.
$$

If one basis had many more samples than another, pooled minibatching would weight basis contributions by empirical sample frequency rather than giving each basis-average equal outer weight. In that case, the implemented loss would no longer be a simple global rescaling of the theoretical objective.

So the pooled loader is theoretically clean **only because the basis shot counts are equal**.

#### What is approximated

The implementation remains approximate in exactly the intended sense:

- the positive multi-basis data term is evaluated exactly on the sampled minibatch
- the amplitude model term is approximated stochastically
- minibatch shuffling introduces the usual stochastic gradient noise

In the original theory text this surrogate is described in CD-$k$ language. The code uses **PCD-$k$** instead. This is not a conceptual mismatch. It is simply a different Monte Carlo estimator for the same intractable amplitude-model expectation.



**Don' Forget:** We use 10% noise replacement and also do Bernoulli noise initially.



#### Final verdict

For equal shot counts per basis, the code implements the Chapter 6 practical objective as a valid stochastic surrogate:

- the positive term is exact in form
- the model term is approximated in the intended RBM fashion
- the pooled loader introduces only a global factor $\frac{1}{R+1}$, not a harmful reweighting
- that global factor can be absorbed into the learning rate

So, as a statement suitable for reference:

**The training logic is theoretically aligned with the practical multi-basis objective, up to the intended stochastic approximations of minibatching and PCD-based estimation of the amplitude model term.**

#### Minor precision notes

- The implementation uses **PCD-$k$** rather than literal fresh-start CD-$k$.
- Any diagnostic based on pooled full-dataset averaging may live on a globally rescaled version of the thesis loss, even when the training objective itself is aligned.
- These points do not affect the theoretical interpretation of the actual training loss.