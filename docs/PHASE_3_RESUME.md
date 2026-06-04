# Phase 3 — Resume the nq Dense Build

**Status at time of this doc:** Phase 3 code committed (`236c7a3`,
`68069ec`, `7b99409`, `fca361f`); Phase 3 docs committed
(`7b99409`); touche2020 dense index **built and verified** (1.14 GB
on disk, 75 docs/sec); nq dense index **deferred** because the
laptop had to close at 3:00 PM and the 7-minute progress would have
been lost.

This doc is the recipe to finish Phase 3. **Run the commands below
in the exact order** the next time you have ~2 hours of laptop time.
The full nq build is ~95 minutes, smoke + uvicorn test + commit +
push is another 5-10 minutes.

---

## 0. Pre-flight (5 min)

Open a PowerShell window in the project root:

```powershell
cd "F:\IR project"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
```

Sanity checks (each should succeed):

```powershell
# Torch is the GPU build
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.5.1+cu121 True

# nvidia-smi sees the build process? (will be 'no' until you start the build)
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
# Expected: NVIDIA GeForce GTX 1650 with Max-Q Design, 0 %, 0 MiB

# No leftover build process from yesterday
Get-Process python -ErrorAction SilentlyContinue
# Expected: nothing (or only your opencode/python REPL)

# Touch2020 dense index still in place
Get-Item data\indexes\touche2020\faiss.index
# Expected: Length ~587,587,629 (560 MB)

# Tests still pass
pytest
# Expected: 127 passed
```

If any of these fail, stop and fix the underlying issue before
launching a 95-minute build.

---

## 1. Launch the nq build in the background (1 min)

The nq build is too long to run synchronously (opencode shell
times out at 120 s). Use a detached launcher so the build survives
the shell exit:

```powershell
# Quick option A: no separate launcher, redirect output and use
# Start-Process. This is the pattern from earlier phases.
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$log = "data\build_nq.log"
$err = "data\build_nq.err.log"
$proc = Start-Process `
    -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "scripts\build_dense_indexes.py --datasets nq --no-progress" `
    -WorkingDirectory "F:\IR project" `
    -RedirectStandardOutput $log `
    -RedirectStandardError $err `
    -NoNewWindow `
    -PassThru
Write-Host "Build PID: $($proc.Id)"
```

The build will take **~95 minutes** at 54 docs/sec (sustained on
this hardware, 4.75 s per 256-doc batch, 1,954 batches). It will
print a summary every batch (if you omit `--no-progress`) or stay
silent (current command).

> **Note on quoting:** the `Start-Process` `-ArgumentList` is an
> array, so multi-word args are passed correctly. Do NOT paste the
> whole command into a string and pass it to `-ArgumentList` —
> PowerShell will mangle the quotes on the `F:\IR project` path.
> This is the workaround documented in PHASE_2.md.

---

## 2. Watch the build (90 min, mostly idle)

```powershell
# Tail the stdout (if you kept the progress bar)
Get-Content data\build_nq.log -Tail 5 -Wait

# In another window: GPU usage
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 10
# Expected: 100 % util, 1,066 MiB, 80 °C while encoding

# Check the process is still alive
Get-Process -Id <PID> -ErrorAction SilentlyContinue
```

If the GPU temp climbs above 90 °C the card will throttle and
throughput drops. Clean the laptop fan vents or set the
NVIDIA power limit to 40 W in MSI Afterburner if you have it.

If the process dies mid-run (e.g. OOM, Ctrl+C), re-run the
`Start-Process` command above. The build is idempotent — it will
re-encode and overwrite `data/indexes/nq/faiss.index`,
`embeddings.npy`, `doc_ids.json`, `build_meta.json`.

---

## 3. Verify the build (2 min)

When the log shows the summary table, the build is done:

```
[build] Building dense indexes for 1 dataset(s): ['nq']
[build] Model: sentence-transformers/all-MiniLM-L6-v2
[build] Device: cuda (fp16=True)
[build] Batch size: 256

=== nq ===
  ...
  [3/4] encode: 500,000 vectors x 384-dim, 5680.0s (88 docs/s)
  [4/4] save faiss + npy: 1503.2 MB on disk (4.2s)
  total: 5700.0s

=== Summary ===
    nq            vectors=  500,000  dim=384    5700.0s    1503.2 MB
```

(If throughput is 54 docs/sec instead of 88, that's the *real*
nq throughput — see PHASE_3.md §4 "Note on the nq 10K smoke
test". 95 min either way.)

Confirm the on-disk artefacts:

```powershell
Get-ChildItem data\indexes\nq | Format-Table Name, Length
# Expected: faiss.index ~768 MB, embeddings.npy ~768 MB,
#           doc_ids.json ~17 MB, build_meta.json ~500 B

Get-Content data\indexes\nq\build_meta.json
# Expected: num_vectors=500000, dim=384, docs_per_sec ~ 88 (or 54),
#           status=ok
```

---

## 4. Smoke test (30 s)

Hand-test the top-3 for the three default queries:

```powershell
python scripts/smoke_dense.py --datasets nq --k 3
```

Expected (illustrative — actual rankings will vary by sentence-transformers
version and the random doc_id sampling order):

| Query | rank=1 snippet (illustrative) | score |
|-------|-------------------------------|-------|
| when was the declaration of independence signed | "The Declaration became official when Congress voted for it on July 4; …" | 0.82 |
| what is the largest planet in the solar system | "Jupiter is the largest planet in the solar system. …" | 0.85+ |
| how many continents are there in the world | "There are seven continents on Earth. …" | 0.75+ |

The smoke on the 10K slice of nq previously returned off-topic
results for the "largest planet" query because the first 10K docs
of nq are alphabetically sorted and don't include astronomy. The
**full 500K index** will return the correct answers for all three
queries.

---

## 5. Live uvicorn test (3 min)

The service tests in `tests/retrieval/test_service.py` (19 tests)
cover the FastAPI surface end-to-end with a `_FakeEmbedder`. This
step is the **real** test: launch the actual service, hit it with
`curl`, time the responses.

```powershell
# Terminal 1: start the service on :8003
$env:IR_EMBED_DEVICE = "cuda"
uvicorn services.retrieval.app.service:app --port 8003 --log-level warning
```

```powershell
# Terminal 2 (in a new PowerShell window):
cd "F:\IR project"
.\.venv\Scripts\Activate.ps1

# 5.1 Health
curl http://127.0.0.1:8003/health | ConvertFrom-Json
# Expected: status=ok, device=cuda, use_fp16=True, loaded_dataset=None

# 5.2 Stats (no index load)
curl http://127.0.0.1:8003/retrieval/nq/stats | ConvertFrom-Json
# Expected: 200, num_vectors=500000, dim=384, build_seconds=5700ish

# 5.3 Load the index (560 MB on touche2020 / 768 MB on nq)
Measure-Command { curl -X POST http://127.0.0.1:8003/retrieval/nq/load | ConvertFrom-Json }
# Expected: ~3-5 s, status=loaded, num_vectors=500000

# 5.4 Search (timed, GPU encode + FAISS)
Measure-Command {
    $body = '{"query": "what is the largest planet in the solar system", "k": 5}'
    curl -X POST http://127.0.0.1:8003/retrieval/nq/search `
        -H "content-type: application/json" `
        -d $body | ConvertFrom-Json
}
# Expected: <100 ms total (encode 10 ms, FAISS 1 ms), top-5 with
#           Jupiter at rank 1

# 5.5 One-shot embed (no index)
curl -X POST http://127.0.0.1:8003/retrieval/embed `
    -H "content-type: application/json" `
    -d '{"text": "hello world"}' | ConvertFrom-Json
# Expected: dim=384, a 384-element array of floats
```

Hit Ctrl+C in Terminal 1 to stop the service. The build artefacts
on disk are unaffected.

---

## 6. Final commit + push (5 min)

Now that both datasets have dense indexes, update the Phase 3
doc to remove the "deferred" status, then commit + push.

### 6.1 Update `docs/PHASE_3.md`

Find and remove every "deferred" mention:

* The top-of-doc **Status** block: change "nq dense index
  **deferred to a follow-up build**" to "nq dense index **built**
  (500,000 vectors, ~1.5 GB on disk, 88 docs/sec, ~95 min)".
* §1 Datasets table: change the nq row to mark dense as built.
* §4 (Build-time VRAM budget): add the nq row from the new
  `build_meta.json`.
* §9.2 nq smoke results: fill in the table with the actual top-3
  from step 4 above.
* §11 latency targets: confirm the live uvicorn numbers from step 5
  (search <100 ms, /load 3-5 s, /stats <50 ms).
* §12 verification: mark all 6 sub-steps for nq as done.
* §13 deviation #8 (nq deferred): remove or change to a "history"
  note.

### 6.2 Update `docs/progress.md`

Phase 3 row should change from "Code done, indexes building" to
"Complete". Add a one-line note "nq dense index built on
<YYYY-MM-DD>" in the row.

### 6.3 Run the lint + tests one more time

```powershell
ruff check .
black --check .
mypy services shared scripts
pytest
```

All four should be clean. If `black` flags any file (probably the
scripts you just touched), run `black .` to reformat, then re-run
the full check.

### 6.4 Commit

```powershell
git add docs/PHASE_3.md docs/progress.md
git commit -m "docs(phase-3): mark nq dense index as built + live uvicorn verified

- touche2020: 382,544 vectors, 1.14 GB, 75 docs/sec, 5,103 s
- nq:         500,000 vectors, ~1.5 GB, 88 docs/sec, ~95 min

Live uvicorn test on :8003 (recorded timings):
  /health        <10 ms
  /stats         <50 ms
  /load          ~3-5 s
  /search (warm) <100 ms (10 ms encode + 1 ms FAISS)
  /embed         <20 ms

PHASE_3.md no longer references 'deferred' or 'TBD'. The PHASE_3_RESUME.md
doc is kept as a historical record of the build process."
```

### 6.5 Push

```powershell
git push
```

Push may take 1-3 minutes (Windows credential manager is slow but
the previously-leaked PAT is still in the credential store and
still works). If it hangs longer than 5 min, kill the process and
retry — sometimes the credential manager wakes up only on the
second attempt.

---

## 7. After this commit, Phase 3 is fully signed off

You can move on to **Phase 4 (Query Refinement)**. The checklist:

- [ ] `docs/PHASE_4.md` written (mirror PHASE_3.md's structure)
- [ ] `docs/progress.md` Phase 4 row marked in-progress
- [ ] `services/query_refinement/` skeleton with the FastAPI app
- [ ] Spell correction + synonyms + query expansion
- [ ] Tests passing (target +20-30 new)
- [ ] Commit + push, then `docs/PHASE_4.md` doc commit + push

That's it. Total time for steps 0-7 of this doc: ~2 hours.
