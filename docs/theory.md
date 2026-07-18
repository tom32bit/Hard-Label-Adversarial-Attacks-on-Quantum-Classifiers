# Theoretical results

Companion proofs for *Hard-Label Adversarial Attacks on Quantum Classifiers*.
Every proposition is stated, proved, and cross-linked to the code/experiment that
confirms it numerically. GitHub renders the math ($\LaTeX$ delimiters).

## Setup and notation

A binary variational quantum classifier on $n$ qubits outputs the decision value

$$
f_\theta(x) \;=\; \langle 0|^{\otimes n}\, E(x)^\dagger W(\theta)^\dagger\, M\, W(\theta)\, E(x)\, |0\rangle^{\otimes n}\;\in[-1,1],
\qquad \hat y(x)=\operatorname{sign} f_\theta(x),
$$

with $E(x)$ the data encoding, $W(\theta)$ the trained ansatz, and $M$ a Hermitian
observable that is **diagonal in the computational basis with eigenvalues in
$\{-1,+1\}$** (e.g. $M=Z_0$ or the global parity $M=Z^{\otimes n}$). Write the
**margin** $m(x)=|f_\theta(x)|$. A query at shot budget $S$ returns a stochastic
label $\hat y_S(x)=\operatorname{sign}\hat f$, where $\hat f$ is the $S$-shot empirical
mean of $M$. The adversary sees only $\hat y_S$ (hard-label) and controls $S$.

Throughout, $\Phi$ is the standard normal CDF, $\varphi$ its density,
$z_\delta=\Phi^{-1}(1-\delta)$, and $\operatorname{erf}$ the error function.

---

## Proposition 1 (the Born-rule oracle is an exact Binomial)

*For a diagonal observable $M$ with spectrum $\{-1,+1\}$ and a state re-prepared each
shot, the $S$-shot measurement is exactly*

$$
k \sim \mathrm{Binomial}\!\Big(S,\ p\Big),\quad p=\tfrac{1+f_\theta(x)}{2},
\qquad
\hat f = \tfrac{2k}{S}-1,\qquad \hat y_S(x)=\operatorname{sign}(2k-S).
$$

*Consequently $\ \mathbb E[\hat f]=f_\theta(x)$, $\ \operatorname{Var}[\hat f]=\dfrac{1-f_\theta(x)^2}{S}$,
and the flip probability relative to the true label is, under the CLT,*

$$
p_{\mathrm{flip}}(x,S)\;=\;\Phi\!\Big(-\,m(x)\sqrt{\tfrac{S}{\,1-f_\theta(x)^2\,}}\Big).
$$

**Proof.** Let $\Pi_+=\tfrac12(I+M)$ be the projector onto the $+1$-eigenspace of $M$.
By the Born rule a single projective measurement returns $+1$ with probability
$\langle\psi|\Pi_+|\psi\rangle=\tfrac12(1+\langle M\rangle)=\tfrac{1+f_\theta(x)}{2}=:p$
and $-1$ otherwise. Because the circuit is re-prepared and measured independently on
each of the $S$ shots, the outcomes are i.i.d. Bernoulli$(p)$, so the count of $+1$
outcomes is $k\sim\mathrm{Binomial}(S,p)$; this $k$ is a sufficient statistic. The
empirical mean of the $\pm1$ eigenvalues is
$\hat f=\big(k-(S-k)\big)/S=2k/S-1$, and $\operatorname{sign}\hat f=\operatorname{sign}(2k-S)$.
Moments: $\mathbb E[k]=Sp\Rightarrow\mathbb E[\hat f]=2p-1=f_\theta(x)$, and
$\operatorname{Var}[k]=Sp(1-p)$ with $p(1-p)=\tfrac{(1+f)(1-f)}{4}=\tfrac{1-f^2}{4}$, so
$\operatorname{Var}[\hat f]=4\operatorname{Var}[k]/S^2=(1-f^2)/S$. For $f>0$ (true label
$+1$) a flip is $\hat f<0$; approximating $\hat f\approx\mathcal N\!\big(f,(1-f^2)/S\big)$
gives $\Pr[\hat f<0]=\Phi\!\big(-f\sqrt{S/(1-f^2)}\big)$, and by symmetry the general
statement uses $m=|f|$. $\qquad\blacksquare$

**Why it matters.** The oracle used by every attack samples one $\mathrm{Binomial}$
draw from the once-computed $f_\theta(x)$ instead of materialising $S$ device shots.
Prop 1 says this is not an approximation but the *exact* measurement law, so the
compute optimisation changes nothing statistically.

**Numerical confirmation.** Sanity test **T1** ([`hlq/oracle.py`](../hlq/oracle.py)
`validate_p_flip`): the CLT $p_{\mathrm{flip}}$ matches the Monte-Carlo Binomial rate
across $m\in[0,1]$, $S\in[10,10^4]$ (max abs. discrepancy $0.017$, only at $S=10$),
and matches PennyLane's own shot sampler within $3$ s.e. Limits recovered:
$S\to\infty\Rightarrow p_{\mathrm{flip}}\to0$; $m\to0\Rightarrow p_{\mathrm{flip}}\to\tfrac12$.

---

## Proposition 2 (shot-limited boundary resolution)

*To decide $\operatorname{sign} f_\theta(x)$ with per-decision error $\le\delta<\tfrac12$
requires*

$$
S \;\ge\; S^\star(f,\delta) \;=\; \big(1-f^2\big)\Big(\tfrac{z_\delta}{|f|}\Big)^{2},
$$

*which diverges as $|f|\to 0$. Equivalently, a per-decision budget $S$ resolves the
boundary only to the **resolution** $\tau(S)=z_\delta\sqrt{(1-f^2)/S}$ in decision-value
space: a point is decided at confidence $1-\delta$ iff $|f|>\tau(S)$. On the boundary
$f=0$ the sign is undecidable at **any** finite $S$.*

**Proof.** By Prop 1, $p_{\mathrm{flip}}=\Phi\!\big(-|f|\sqrt{S/(1-f^2)}\big)$. Since
$\Phi$ is increasing, $p_{\mathrm{flip}}\le\delta \iff |f|\sqrt{S/(1-f^2)}\ge
\Phi^{-1}(1-\delta)=z_\delta \iff S\ge (1-f^2)z_\delta^2/f^2$. As $|f|\to0$ the bound
$\to\infty$. Rearranging the threshold $|f|\ge z_\delta\sqrt{(1-f^2)/S}=\tau(S)$ gives
the resolution form; at $f=0$, $p_{\mathrm{flip}}=\Phi(0)=\tfrac12$ for every $S$.
$\qquad\blacksquare$

### Corollary 2.1 (the naive fixed-shot port diverges with budget)

*Consider a binary search that bisects to angular tolerance $\theta$
($K=\lceil\log_2(1/\theta)\rceil$ steps) using a fixed shot count $S$, treating the
returned high-endpoint as adversarial. As the search approaches the true boundary the
midpoint margins $\to0$, so the per-decision error $\to\tfrac12$ (Prop 2), and a false
positive is absorbing (the high-endpoint only moves toward $x_0$). Hence*

$$
\Pr[\text{returned point is truly adversarial}]\ \longrightarrow\ 0
\quad\text{as }\theta\to0,
$$

*and spending more total budget on a finer search makes the naive attack strictly
worse.*

**Proof sketch.** Near the boundary a fixed $S$ cannot hold error below $\delta$
(Prop 2 requires $S\to\infty$), so the last $\Omega(1)$ bisection decisions are
essentially fair coins. One erroneous "adversarial" verdict sets the high-endpoint
strictly inside the true class and is never undone, so the terminal point has negative
true margin with probability $\to 1$ as the number of near-boundary decisions grows
with $K$. $\qquad\blacksquare$

**This is the mechanism the calibrated attack (M1) avoids:** it *stops* bisecting at
$\tau(S)$ rather than pushing into the coin-flip zone, so every accepted boundary point
carries margin $\ge\tau>0$ and survives ground-truth verification.

**Numerical confirmation.** Corollary 2.1 predicts the RQ3 signature exactly: the
fixed-shot port's verified success **falls** with budget, $0.36\to0.14\to0.07$ at
$T=15\text{k},60\text{k},120\text{k}$ ([`results/rq3.json`](../results/rq3.json)),
while the calibrated attack, which respects $\tau(S)$, holds $0.70\to0.59\to0.58$.
Sanity test **T2** shows the calibrated attack $\to$ deterministic HopSkipJump as
$S\to\infty$ (paired per-image gap $\to 4\%$).

---

## Proposition 3 (optimal shot/probe split — the query/shot economics)

At a boundary point the normal is estimated by
$\hat g=\frac1B\sum_{i=1}^{B}\phi_i u_i$ with i.i.d. unit probes $u_i$ and noisy
adversarial indicators $\phi_i\in\{\pm1\}$, each measured with $S$ shots at typical
probe margin $m$. A flip attenuates the signal, $\mathbb E[\phi_i\mid u_i]=(1-2p)\,
\sigma_i$ with $p=p_{\mathrm{flip}}(m,S)$, so the direction estimate has

$$
\mathrm{SNR}^2 \;\propto\; B\,(1-2p)^2,
\qquad\text{and under a fixed gradient budget }T_{\mathrm g}=B\,S,\quad
\mathrm{SNR}^2 \;\propto\; T_{\mathrm g}\,\underbrace{\frac{\big(1-2p_{\mathrm{flip}}(m,S)\big)^2}{S}}_{=:~h(S)}.
$$

**Claim (a) — the gradient sub-problem has no interior optimum.** *For every fixed
$m\in(0,1)$, $h$ is strictly decreasing on $S>0$. Hence the SNR-optimal allocation is
the minimum shots / maximum probes, $S^\star_{\mathrm{probe}}=1$: a boundary optimum,
not an interior one. In the noise-dominated small-$S$ regime
$h(S)\to \tfrac{2m^2}{\pi(1-m^2)}$, independent of $S$ — a plateau.*

**Proof.** Using $1-2\Phi(-x)=\operatorname{erf}(x/\sqrt2)$ with
$x=m\sqrt{S/(1-m^2)}$, set $c=\tfrac{m}{\sqrt{2(1-m^2)}}$ and $y=c\sqrt S$. Then
$h(S)=\operatorname{erf}(y)^2/S=c^2\big(\operatorname{erf}(y)/y\big)^2$. It suffices
that $y\mapsto \operatorname{erf}(y)/y$ is strictly decreasing on $y>0$:

$$
\frac{d}{dy}\frac{\operatorname{erf}(y)}{y}
=\frac{\operatorname{erf}'(y)\,y-\operatorname{erf}(y)}{y^2}
=\frac{1}{y^2}\Big(\tfrac{2}{\sqrt\pi}y\,e^{-y^2}-\operatorname{erf}(y)\Big)=:\frac{b(y)}{y^2}.
$$

Now $b(0)=0$ and
$b'(y)=\tfrac{2}{\sqrt\pi}\big(e^{-y^2}-2y^2e^{-y^2}\big)-\tfrac{2}{\sqrt\pi}e^{-y^2}
=-\tfrac{4}{\sqrt\pi}y^2e^{-y^2}<0$ for $y>0$, so $b(y)<0$ on $y>0$; therefore
$\operatorname{erf}(y)/y$ is strictly decreasing, $h$ is strictly decreasing in $y$
hence in $S$, and $\lim_{y\to0^+}\big(\operatorname{erf}(y)/y\big)^2=(2/\sqrt\pi)^2=4/\pi$
gives $h(S)\to 4c^2/\pi=\tfrac{2m^2}{\pi(1-m^2)}$. The exact Binomial makes small $S$
even more favourable: at $S=1$, $1-2p_{\mathrm{flip}}=m$ so $h(1)=m^2$, exceeding the CLT
plateau value. $\qquad\blacksquare$

**Claim (b) — the interior optimum lives in the M1/M2 split.** *Boundary-localisation
shots (M1) must grow like $S^\star(f,\delta)=\Theta(1/m^2)$ near the boundary (Prop 2),
whereas gradient shots (M2) are best minimised (Claim a). Splitting a fixed
per-iteration budget between the two therefore trades a term increasing in the
M1 share against a term decreasing in it, producing an interior optimum in the split
fraction — not in the probe-shot count.*

**Consequence for H2 (and RQ2).** The plan's H2 conjectured an interior optimum in
the query/shot split. Prop 3 makes this **precise and corrects it**: the probe-shot
allocation optimum is on the boundary $S_{\mathrm{probe}}=1$ (H2 is *false* for the
gradient sub-problem — a clean, predicted negative result), while the genuine interior
optimum is the **M1-vs-M2 budget division**. This converts a shallow empirical dip into
a theorem that says *where* to look.

**Numerical confirmation.** [`hlq/budget.py`](../hlq/budget.py) `grad_snr_per_budget`
is exactly $h(S)$; the RQ2 allocation sweep bottoms out at the smallest shots
($S_{\mathrm{probe}}\!\approx\!1$–$2$) with a shallow, within-noise profile — consistent
with the proven plateau — while the M1/M2 split shows a clearer interior minimum at
$\approx0.75$ ([`results/rq2.json`](../results/rq2.json)). Fig. `fig_RQ2_allocation`
overlays the closed-form $h(S)$.

---

## Theorem 4 (concentration ⟹ exponential hard-label query complexity)

*Suppose the effective decision-value margins under a defense concentrate: there are
constants $C>0$, $b>1$ such that a $1-o(1)$ fraction of test inputs have
$m(x)\le C\,b^{-n/2}$ (exponential concentration, cf. barren plateaus / Thanasilp et
al.). Then any hard-label attacker — regardless of strategy — needs*

$$
S \;\ge\; S^\star(m,\delta)\;=\;\Omega\!\big(b^{\,n}\big)\ \text{shots per boundary decision}
$$

*to maintain per-decision error $\le\delta<\tfrac12$, hence $\Omega(b^{n})$ total
measurement budget for the $\Theta(\mathrm{poly})$ reliable decisions a boundary walk
needs. The bound is unconditional (information-theoretic), not an artefact of a
particular attack.*

**Proof.** For any input in the concentrated bulk, $m\le Cb^{-n/2}$ gives, by Prop 2,
$S^\star=(1-m^2)z_\delta^2/m^2\ge z_\delta^2\,(1-m^2)/(C^2 b^{-n})=\Omega(b^{n})$. This
is a property of the *label channel* (a $\mathrm{Bernoulli}(p_{\mathrm{flip}})$ with
$p_{\mathrm{flip}}\to\tfrac12$), so it lower-bounds every attacker that only observes
labels: the mutual information between one $S$-shot label and the true side of the
boundary is $1-H_2(p_{\mathrm{flip}})=O(m^2 S)$, so $\Omega(1/m^2)=\Omega(b^n)$ shots
are needed to acquire one bit. Summing over the $\Theta(\mathrm{poly})$ boundary
decisions gives the total. $\qquad\blacksquare$

**Interpretation — genuine defense vs. gradient masking.** Classical gradient
masking breaks white-box gradients while a *gradient-free* attack still succeeds
cheaply. Theorem 4 shows the opposite for concentration-based encodings: the
**gradient-free, shot-limited** attacker (the realistic one) provably pays an
exponential price, tied to a real information-theoretic barrier — a non-masking
protection, the asymmetry the plan (H4b) sought. **But the guardrail is essential:**
the same concentration that raises attack cost also collapses clean accuracy toward
chance unless the encoding preserves *data-relevant* margins while suppressing only
*attack-relevant* ones; a defense that concentrates everything is merely "trivially
robust" (Sec. RQ5), not secure.

**Numerical confirmation.** RQ4 at $n=4$ ([`results/rq4.json`](../results/rq4.json))
realises both regimes: the **randomized-encoding** defense drives verified success
$0.80\!\to\!0.12$ while keeping clean accuracy $0.83$ (the Theorem-4 barrier with
accuracy preserved), whereas the **depolarizing** defense drives success $0.80\!\to\!0.40$
but collapses accuracy to $0.48<\tfrac12$ (trivially robust — the guardrail fires).
The exponential-in-$n$ scaling of $S^\star$ is the prediction RQ4-across-$n$ tests.

---

## Summary of what the theory settles

| # | Result | Settles |
|---|--------|---------|
| Prop 1 | exact Binomial oracle + $p_{\mathrm{flip}}$ closed form | the oracle model (T1); validates the fast analytic path |
| Prop 2 | resolution $\tau(S)$; naive port diverges with budget | the M1 design; **explains RQ3's fixed-shot decay** |
| Prop 3 | gradient SNR $h(S)$ strictly decreasing; interior optimum is in the M1/M2 split | **reframes RQ2/H2** into a predicted plateau + a located interior optimum |
| Thm 4 | concentration $\Rightarrow \Omega(b^n)$ hard-label cost | **explains RQ4's randomized-encoding defense**; ties it to the RQ5 guardrail |
