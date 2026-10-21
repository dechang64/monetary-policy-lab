# Appendix C — Proofs and Derivations

## C.1 Proposition 1 (inversion)

**Setup.** Investors observe the decision instrument $a$ (a futures-implied surprise) and the window price $p$. Let $\theta$ be central-bank private information (bad news = high), $a = \alpha m$ with $m \sim N(0, \tau_m^{-1})$ the pure action innovation, and let the information channel load on $p$ with positive sign when $\theta$ is good: $p = \phi c + \gamma m + u$, $c \sim N(0, \tau_c^{-1})$, $\text{corr}(c, \theta) > 0$, $u$ iid noise.

**Claim.** Classifying a target-shock event $s = a$-innovation as $m$ when $\text{sign}(s) \neq \text{sign}(\Delta p_{\text{market}})$ and as $c$ when $\text{sign}(s) = \text{sign}(\Delta p_{\text{market}})$ recovers the decomposition up to scale, with error rate equal to $P(u \text{ flips the sign}) = 2\Phi(-|\text{SNR}|)$, where SNR is the signal-to-noise ratio of the window price.

**Proof.** Conditional on an action event ($c = 0$): $\Delta p = \gamma m + u$ has the same sign as $m$ with probability $1 - 2\Phi(-|\gamma m|/\sigma_u) > 1/2$, so the rule assigns $m$ correctly with that probability. Conditional on an information event with an associated action ($m = a/c$ scale): $\Delta p = \phi c + \gamma m + u$; for $\phi > |\gamma|$ (information dominates in the market's posterior, as in JK's calibration), the sign of $\Delta p$ follows $c$ with probability $1 - 2\Phi(-|\phi c + \gamma m|/\sigma_u) > 1/2$. Summing, the classification error is exactly the sign-flip probability of the noise, which is minimized when the two components' price loadings are most distinct — the JK condition. $\square$

*Correspondence to the implementation.* The Section 3.1 rule uses the same-day CRSP sign as $\Delta p_{\text{market}}$ and the Acosta target-shock as $s$; the JK calibration $\phi > |\gamma|$ is what makes the rule an inversion rather than a heuristic.

## C.2 Proposition 2 (regime-dependent text relevance)

**Setup.** Tone $\hat t = t + e$, $e \sim N(0, \tau_e^{-1})$, $t = \beta_t \theta + \beta_a m + \varepsilon_t$ with paradigm-dependent informativeness $\beta_t$: under forward guidance $\beta_t \neq 0$ (words describe the state and the reaction function); in action-centric paradigms $\beta_t = 0$.

**Claim.** $\partial \mathbb{E}[\theta \mid p, \hat t] / \partial \hat t = 0$ iff $\beta_t = 0$; and the FG-regime coefficient on tone in a regression of $\Delta p$ on $(m, c, \hat t)$ equals $\beta_t$ times the posterior weight on $\theta$, positive in rates effects and negative in equity (discount-rate and cash-flow channels).

**Proof.** Bayesian updating is linear in $(p, \hat t)$; the posterior weight on $\hat t$ is zero iff its likelihood is uninformative, i.e., $\beta_t = 0$. Given $\beta_t \neq 0$, the projection of $\Delta p$ on $(m, c, \hat t)$ loads on $\hat t$ with the sign of the $\theta$-channel times the tone-$\theta$ loading; the rates channel moves opposite the equity channel by the standard discount-rate identity (higher expected path ⇒ lower price), producing the Table 4.1 sign pattern. $\square$

## C.3 Proposition 3 (flows as residual demand)

**Setup.** Investor type $k$ holds factor portfolio with risk-rank $r_k \in \{1..7\}$; mean-variance demand with risk price $\lambda$: $w_k^* \propto \lambda^{-1} \Sigma^{-1} \mu_k$, where the information update $c$ changes expected cash-flow growth $\mu_k$ by $-b_k c$ with loading $b_k$ increasing in $r_k$ (risky claims carry more of $\theta$). Monthly flow $f_k = \Delta w_k^*$ measured after the price $p$ has cleared.

**Claim.** (i) $f_k \propto -r_k c$ (the ladder); (ii) $f_k \perp r_k$ for pure action news $m$ that changes only the short rate; (iii) $|f_k|$ is an order of magnitude below the price response at the monthly horizon.

**Proof.** (i) $\mu_k$ shifts by $-b_k c$ with $b_k$ monotone in $r_k$, so $\Delta w_k^* \propto -\lambda^{-1} \Sigma^{-1} b_k c$, and under a single-factor loading approximation $b_k = b_1 r_k$ the reallocation is proportional to $-r_k c$ — the empirical Risk × CBI. (ii) An action news $m$ moves the short-end price only; with cash-flow expectations $\mu_k$ unchanged, $w_k^*$ moves by the (class-invariant) discount-rate term, so the *cross-sectional* ladder coefficient is zero even though the level term is not — matching Risk × MP = 0 with the ZLB exception, where the action's rescue-capacity content re-enters $\mu$. (iii) Because $p$ already embeds $c$ at the announcement, the monthly $\Delta w_k^*$ reflects only the residual rebalancing toward the new optimum — the timing structure that generates T3. $\square$

*Empirical mapping.* (i) → Table 6.1 (Risk × CBI = −0.066, p = 0.036); (ii) → Risk × MP = +0.005 (p = 0.88) with the MP × ZLB exception (−1.505, p = 0.005); (iii) → the Section 5 nulls.
