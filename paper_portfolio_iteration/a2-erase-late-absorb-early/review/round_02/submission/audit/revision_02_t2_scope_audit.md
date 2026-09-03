# Revision 02 T2 scope and proof audit

Date: 2026-08-22. This audit addresses Round-1 issue `TECH-T2-001` without
changing the frozen Round-1 snapshot.

## Printed premise

For every actual gradient-descent iterate `t=s,...,T-1` and every point
`r in [0,1]` on its segment from the minimizer, T2 now prints

\[
\mu I \preceq \nabla^2 L_A(w_A^\star+r(w_t-w_A^\star)) \preceq LI.
\]

This is a pointwise ambient Loewner-order bound, not a directional curvature
condition, a restricted scalar statement along the segment, or an ambiguous
use of ``strongly convex and smooth on a segment.''

## Verification

The fundamental theorem of calculus gives

\[
\nabla L_A(w_t)-\nabla L_A(w_A^\star)=\bar H_t(w_t-w_A^\star),\quad
\bar H_t=\int_0^1\nabla^2L_A(w_A^\star+r(w_t-w_A^\star))\,dr.
\]

Integration preserves the printed PSD inequalities, so
`mu I <= Hbar_t <= L I`. For `0 <= eta_t <= 1/L`, every eigenvalue of
`I-eta_t Hbar_t` lies in `[0, 1-eta_t mu]`; its spectral norm is therefore at
most `1-eta_t mu`. Multiplication yields the full-norm product bound and
`1-x <= exp(-x)` yields the exponential bound. The directional line remains
conditional on simultaneous invariance of `span(e)` for every `Hbar_t` and a
positive eigenvalue lower bound `lambda_e`.

## Counterexample exclusion

The Round-1 off-segment quadratic counterexample is not admitted by the new
premise: its large cross term makes the ambient upper bound on the full Hessian
fail for the proposed small `L`. Thus the revised printed premise is exactly
the premise used by the full-norm proof.

## Scope outcome

T2 contracts an arbitrary initial iterate `w_s` to the minimizer of its one
displayed objective. It contains no intervention variable, comparator path, or
bound on an intervened-versus-reference difference. Revision 02 consequently
does not call it a clean-tail return or an unlearning identity.
