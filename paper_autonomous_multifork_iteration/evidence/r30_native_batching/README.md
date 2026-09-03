# R30 native batching evidence

This directory is an affected-only experiment requested after PDF review.  It
observes the real vLLM 0.26 V1 scheduler on one fixed Qwen3.5-35B-A3B / H20-3e
stack.  Three distinct PG19 requests are queued with `max_num_seqs=2`: ragged
requests A/B must enter one `SchedulerOutput`, A must finish first, and waiting
C must enter while B remains live.  Sequential A/B/C controls use the same
loaded engine and greedy sampling.  Full-vocabulary output log-probabilities
are frozen as NumPy sidecars.

The observer is deliberately narrower than ForkAudit.  The existing CoMem /
ForkAudit adapter is a Transformers in-process cache facade and is not wired
into vLLM EngineCore.  Therefore this run can establish native scheduler
batching and scheduler-visible KV block ownership/lifecycle, but it cannot
establish native-batched ForkAudit GDN receipts or production safety.  No
result is automatically manuscript-eligible merely because this run exists.

The QS node is job 249885 / trial 1898483, pod
`qs-249885-1898483-ai-1442658-master-0`.  GPU1 is the only authorized device.
The node must not be stopped by this experiment.
