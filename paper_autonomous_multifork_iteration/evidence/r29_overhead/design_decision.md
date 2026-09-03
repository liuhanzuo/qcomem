# Round 29 overhead design decision

The practicality request is split into independently interpretable estimands.
This preregistration covers the already-shippable reviewer package: exact
logical footprint and CPU-only offline replay time.  A separate H20
preregistration will cover paired live execution with and without receipt
capture after the callable and synchronization boundary are fixed.  The two
measurements will not be merged into one percentage.

The local replay benchmark uses one unreported warmup followed by five measured
runs because the reviewer workflow normally starts from an already unpacked
package.  It is therefore explicitly a warm-cache result.  The benchmark does
not clear the OS page cache, does not include download or extraction, and does
not reinterpret the 852 MiB package as device-memory overhead.

