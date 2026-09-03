# Preregistered Marconi policy trace

This directory is the formal RW-B4 policy-simulator result executed inside the
persistent 8xH20 debug node (QuickSilver Job 246593 / Trial 1871681).  The
simulator is CPU-only; no model inference or GPU timing was rerun.

- preregistration:
  `../marconi_policy_preregistration.json`, SHA-256
  `501448f8326ed5de97fe5378250f48220f20227b3261df626a3e6e705f24514a`;
- runner SHA-256:
  `a3728696aecf92135e8a24924a1dc03b393107bc5599fa036e31bce9fdd75109`;
- official Marconi commit:
  `08016617b1524e6bf6ac29b680641cc945bda7f0`;
- formal summary SHA-256:
  `afb9c6d41973ab3259fc7e50441a6822cba5290a046ed7c237490d67dfe643ec`;
- formal trace SHA-256:
  `536c676dec6cad496ec2825ef0f38e6da9e0dc64f543f0e4615db78e3eb9157f`.

The trace is a disclosed 128-event synthetic multi-turn extension of the eight
frozen Qasper/2WikiMQA document-question prefixes and their already measured
answers.  It uses deterministic counts `(32,24,20,16,12,10,8,6)`, the fixed
follow-up suffix recorded in `trace.json`, and 5/10/20 decimal-GB budgets.  The
formal scientific rows match debug D exactly after excluding context-only
simulator wall time.

Only token-hit rate is reported.  The artifact retains native
Attention--Mamba2 geometry and uses a disclosed post-insert call to each
policy's own eviction routine to close approximate accounting to the exact
byte budget.  These results are not Qwen3.5 serving throughput, LongBench
quality, or a direct CoMem comparison and must not be pooled with the matched
H20 serving table.
