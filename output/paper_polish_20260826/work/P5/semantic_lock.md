# P5 Semantic Lock

- Intervention: 12 epochs; 150/600 affected examples; labels change only at epochs 9--11; deterministic map `y -> (y+1) mod K`.
- Gate statistic: `m = L_11 - min_t L_t`; threshold `tau = 0.1354038956`, frozen on four FIT seeds per mode.
- Hold-out: 64 healthy and 64 shocked seeds, disjoint from FIT seeds.
- Alarm readout: shocked recall 64/64, lower 0.9543; healthy FAR 0/64, upper 0.0457.
- Paired shocked-arm effects: gate-minus-last -0.3395, CI [-0.3643,-0.3189]; last-minus-best-val +0.3395, CI [0.3187,0.3634].
- Selector equality: gate, best-val, and early-stop are outcome-identical on all 64 shocked records and use epoch 8.
- Claim boundary: evidence supports detection plus rollback versus `last`, not checkpoint novelty or superiority to best-val/early-stop.
- FAR boundary: healthy `m=0` follows 704/704 strict adjacent decreases; this is not realistic deployment calibration.
- All tables, figures, equations, citation keys, labels, run paths, and appendix statistics are locked.
