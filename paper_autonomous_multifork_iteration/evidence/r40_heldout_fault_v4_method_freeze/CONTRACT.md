# Method-v4 contract

The verifier/launcher accepts no caller-selected method root. Before import, an
external supervisor must compare the archive bytes and embedded `source-ledger.json`
against the two non-null values in a separately published formal binding. The archive
cannot create or replace that binding. The binding also pins the exact runner-tree
manifest, one literal command vector, formal configuration, physical GPU UUID order,
and exact `NVIDIA H20-3e` SKU.

Every tree is closed-world: regular files only; no symlink, hardlink, duplicate inode,
missing or extra member. Publication is atomic and no-replace. The retained campaign
parent dirfd/inode is checked after every setup stage and at finalization.

The finalizer is installed before locks or output setup. Spawn and process registration
are one critical section. A process remains registered until death is confirmed. Kill
errors are accumulated while all eight failure terminals are attempted. Any lane,
worker, kill, post-rehash, inventory, receipt, or verification failure produces a
nonzero launcher exit. Success requires exactly 8 successful final terminals and the
exact binding/terminal-only tree.

The audited runner itself constructs the fixed CUDA capture backend, synchronizes,
reads live tensors, records the physical UUID reported by torch/NVML, and enforces
H0 peak=current. Callbacks cannot report receipts or allocator/tensor state.

Freeze time is the constant `2026-08-27T00:00:00Z`; tar uid/gid/mtime are 0 and gzip
mtime is 0. The public designer snapshot is created only after the local guard suite
passes. No v4 fault set exists in this package.
