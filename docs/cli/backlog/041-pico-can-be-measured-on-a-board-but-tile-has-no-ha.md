id: 41
area: run
title: PICO can be measured on a board, but tile has no harness for it
opened: 2026-09-28

## wanted
`tile <kernel> -t pico -o k.pico.s -r` to lower, build a container and time it,
the way `-r` does for Metal. PICO is listed "unmeasured — emits correctly; never
run on the hardware", and that is now avoidable: an SS928 board is reachable and
a repo-generated container executes on it.

## got
tile: note: pico targets pico but this machine is apple-gpu — this is cross
  generation, and there is no harness for it here, so the result cannot be run
  or measured. Pass --cross pico to say so deliberately.

## workaround
The loop exists outside `tile`, in the sibling tile-rs-pico checkout, and closes:

  scripts/pico_wrap.py            op program -> native schedule -> words
                                  `--verify` is byte-exact 5/5 against deployed
                                  MiniCPM5 projections, from shape alone
  scripts/pico_wrap.py --into D.om  wrap words into a donor container
  scripts/board/pxr1.py           run any .om on the board, hash outputs, and
                                  time it as write/wait/read (added 2026-09-28)

Measured this way: `build/up_proj_from_tilers.om`, a repo-generated gemm
container (K=1536, N=168, 544 words), runs at WAIT 0.27 ms, deterministic. The
repo's cost model puts the fixed launch floor at ~0.235 ms, so the kernel itself
is ~0.035 ms and the launch dominates — which a harness would surface directly.

What the fix will need:
* the board is reached as `eulerpi-a`, NOT `euler-via-us-vps` (the tunnel moved
  to VPS port 50000);
* nothing PICO runs until `rmmod ot_pqp && insmod /opt/ko/svp_npu/ot_svp_npu.ko`
  — the SS928 has two NPU cores and the board boots the other one (`ot_npu_*`);
* the executor is `pico_persistent_acl_executor.resident.aarch64` + `libsvp_acl.so`,
  speaking a pipe protocol pxr1.py implements;
* time `wait` (flush -> response header), not the whole call: an execute pushes
  every input through a pipe first, 15.1 MB for LRStereo-B, which would swamp it.

SECOND GAP, same area: there is no autotuning surface for PICO. svp's native
schedulers take shape alone — `emit_gemm_instruction_words(K, N)` — and emit one
fixed schedule, so `-O0..-O3` are all identical for this target (confirmed by
docs/evidence/tile-cli-optimize-sweep.tsv) and there is nothing for a search to
vary. Any PICO autotuning has to come from a tile-rs-side scheduler, not from
asking svp's for alternatives.
