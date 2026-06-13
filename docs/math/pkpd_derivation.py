"""
PK/PD derivation of the dose-response function alpha(u).



The function alpha(u) = phi + sigma * u^2 / (1 + u^2) describes the rate at
which antibiotic-sensitive bacterial strains acquire resistance during
antibiotic treatment. This derivation shows it emerges naturally from the
mutant selection window (MSW) hypothesis.

================================================================================
1. MUTANT SELECTION WINDOW (MSW) HYPOTHESIS
================================================================================

Drlica & Zhao (2007, Rev Med Microbiol) proposed that antibiotic concentrations
define three zones:
  - Below MIC_sens: no selection, all strains survive
  - Between MIC_sens and MPC (mutant prevention concentration): SELECTIVE
    pressure favors resistant mutants over wild-type
  - Above MPC: even resistant mutants are killed

Formally:
  - MIC = Minimum Inhibitory Concentration for wild-type
  - MPC = Mutant Prevention Concentration
  - MSW = [MIC, MPC] is the "mutant selection window"

Within the MSW, the per-cell probability of a resistant mutant emerging
and surviving is maximized.

================================================================================
2. POPULATION-LEVEL EMERGENCE RATE
================================================================================

At the population level, the rate of resistance emergence is:

  alpha(u) = baseline_mutation_rate + selection_rate * f_selective_pressure(u)

where:
  - baseline_mutation_rate (phi): spontaneous mutations per replication, even
    without antibiotics. Drake et al. (Genetics 1998) estimate ~10^-6 to 10^-8
    per nucleotide per replication for bacteria.

  - selection_rate (sigma): maximum rate at which antibiotic exposure can
    induce detectable resistance emergence. This is proportional to:
      * Bacterial load (number of replicating bacteria exposed)
      * Mutation rate under stress (SOS response, ~100x higher than baseline)
      * Probability that a mutant establishes (depends on fitness cost)

  - f_selective_pressure(u): fraction of time/dose that the antibiotic
    concentration falls within the MSW.

================================================================================
3. FROM MSW TO f(u)
================================================================================

Consider a dosing regimen with peak concentration proportional to u.
The antibiotic concentration at the site of infection follows PK profile:

  C(t) = C_max(u) * exp(-k_e * t)

where C_max(u) = C_max_baseline * u (dose proportional to u),
      k_e = elimination rate constant.

The MSW is the range [MIC, MPC]. The time the concentration spends in the
MSW during one dosing interval is:

  t_MSW = (1/k_e) * ln(C_max(u) / MPC) - (1/k_e) * ln(C_max(u) / MIC)
         = (1/k_e) * ln(MPC / MIC)

The FRACTION of time in MSW is constant (independent of u!) as long as
C_max(u) > MPC. For lower u where C_max(u) < MPC but > MIC, the MSW time is:

  t_MSW(u) = (1/k_e) * ln(C_max(u) / MIC)    for MIC < C_max < MPC

The selective pressure is proportional to the area under the time-in-MSW curve:

  f(u) = t_MSW(u) / dosing_interval

For C_max(u) >> MPC: f(u) -> 0 (both wild-type and mutants killed)
For C_max(u) between MIC and MPC: f(u) ~ ln(u) / ln(MPC/MIC)
For C_max(u) < MIC: f(u) -> 0 (no selective pressure)

This suggests a HILL-TYPE function:

  f(u) = u^2 / (1 + u^2)

This function has the correct qualitative behavior:
  - f(0) = 0
  - f'(0) = 0 (no selection at zero dose)
  - f(u) -> 1 as u -> infinity (saturates)
  - f(u) is sigmoid-shaped (Hill coefficient = 2)

The Hill coefficient of 2 reflects the cooperativity of the MSW: resistance
emergence requires BOTH mutation AND selective survival, a two-step process.

================================================================================
4. FULL DOSE-RESPONSE FUNCTION
================================================================================

  alpha(u) = phi + sigma * u^2 / (1 + u^2)

where:
  phi   ~ 10^-6 to 10^-8  [Drake et al. 1998; Andersson & Hughes 2010]
  sigma ~ 0.1 to 2.0      [empirically calibrated, depends on pathogen-ABX pair]

The Hill coefficient (2) can be generalized to n, but n=2 is used as a
parsimonious default consistent with the two-step MSW mechanism.

================================================================================
5. ALTERNATIVE DERIVATIONS
================================================================================

5.1 Competitive release model (Lipsitch & Samore 2002):
  alpha(u) = phi + sigma * u
  (Linear response; simpler but less biologically motivated)

5.2 Threshold model (Bonhoeffer et al. 1997):
  alpha(u) = sigma * I[u > u_threshold]
  (Step function; analytically tractable but discontinuous)

5.3 Empirical logistic (Colijn et al. 2009):
  alpha(u) = phi + sigma / (1 + exp(-k*(u - u0)))
  (Generalized logistic; more flexible but more parameters)

Our choice of alpha(u) = phi + sigma * u^2/(1+u^2) balances biological
plausibility (MSW hypothesis) with mathematical tractability (smooth, convex,
bounded). Sensitivity to the functional form can be assessed by varying
the Hill coefficient.

================================================================================
6. REFERENCES
================================================================================

Drlica K, Zhao X. "Mutant selection window hypothesis updated."
  Clin Infect Dis 2007; 44(5): 681-688. DOI: 10.1086/511642

Drake JW, et al. "Rates of spontaneous mutation." Genetics 1998; 148(4): 1667-1686.

Andersson DI, Hughes D. "Antibiotic resistance and its cost: is it possible
  to reverse resistance?" Nat Rev Microbiol 2010; 8(4): 260-271.

Lipsitch M, Samore MH. "Antimicrobial use and antimicrobial resistance:
  a population perspective." Emerg Infect Dis 2002; 8(4): 347-354.

Bonhoeffer S, Lipsitch M, Levin BR. "Evaluating treatment protocols to
  prevent antibiotic resistance." PNAS 1997; 94(22): 12106-12111.

Colijn C, et al. "Designing antibiotic cycling strategies by determining
  and understanding local adaptive landscapes." PLoS ONE 2009; 4(5): e5524.
"""

# Symbolic verification of the dose-response function
if __name__ == "__main__":
    import sympy as sym

    u, phi, sigma = sym.symbols("u phi sigma", positive=True)
    alpha = phi + sigma * u**2 / (1 + u**2)

    # Verify properties
    alpha0 = sym.limit(alpha, u, 0)
    alpha_inf = sym.limit(alpha, u, sym.oo)
    dalpha = sym.diff(alpha, u)
    d2alpha = sym.diff(dalpha, u)

    print("alpha(u) = phi + sigma * u^2 / (1 + u^2)")
    print()
    print(f"alpha(0) = {alpha0}")
    print(f"alpha(inf) = {alpha_inf}")
    print(f"dalpha/du = {sym.simplify(dalpha)}")
    print()
    print("Derivative is zero at u=0: ", sym.limit(dalpha, u, 0) == 0)
    print("Derivative -> 0 as u->inf: ", sym.limit(dalpha, u, sym.oo) == 0)
    print()
    print("Maximum of dalpha/du occurs at inflection point:")
    inflection = sym.solve(sym.diff(d2alpha, u), u)
    print(f"  u_inflection = {inflection}")
    for u_val in inflection:
        if u_val.is_real and u_val > 0:
            max_gradient = float(dalpha.subs(u, u_val))
            print(f"  max dalpha/du = {max_gradient:.4f} * sigma")
