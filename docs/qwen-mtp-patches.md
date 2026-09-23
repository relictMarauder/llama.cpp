# qwen-mtp

`qwen-mtp` is a downstream `llama.cpp` branch focused on reliable native MTP inference for Qwen3.8 on CUDA, including long-context and mixed CPU/GPU tensor placement.

The branch is intentionally maintained as a small linear patch stack on top of upstream `master`.

Current structure:

```text
upstream/master
    |
    +-- ggml : reuse compute buffers for MTP
    |
    +-- common : synchronize MTP draft before target compute reuse
    |
    +-- kv-cache : honor LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY in state_write / state_read_sinfo
    |
    +-- server : share checkpoint state and harden prompt cache OOM handling
```

Do not squash these commits unless there is a specific reason to do so. Keeping them separate makes upstream rebases, regression testing, and eventual removal of patches much easier.

---

## Purpose

The main target is native MTP inference with configurations similar to:

```text
Qwen3.8-27B
CUDA
single sequence
long context up to 65536
mixed CPU/GPU tensor placement
native draft-mtp
shared target/draft compute buffers
prompt checkpoint/cache enabled
```

The branch addresses four related areas:

1. reduce duplicate GPU compute-buffer allocation between target and MTP contexts;
2. make shared compute-buffer reuse safe with asynchronous execution;
3. preserve partial-only KV state semantics;
4. improve server prompt-checkpoint state sharing and cache/OOM handling.

---

# Patch stack

## 1. Adapted PR #27489

Upstream PR:

```text
#27489
ggml : reuse compute buffers for MTP (#27282)
```

Original purpose:

Native MTP creates separate target and draft/MTP schedulers.

Before this patch, each scheduler owns a separate physical compute workspace. Near the VRAM limit this can waste a significant amount of GPU memory and may cause the MTP context to fail allocation even though target and MTP execution are intended to be sequential.

The patch keeps:

```text
independent scheduler state
independent graph allocation state
```

while allowing both schedulers to share:

```text
physical compute buffers
```

The important API contract is conceptually:

```cpp
// Share physical compute buffers while keeping scheduler and graph
// allocation state independent.
//
// The caller must ensure that the schedulers do not execute concurrently
// while the buffers are shared.
bool ggml_backend_sched_share_compute_buffers(...);
```

### Local adaptation

The original #27489 was based on an older `llama.cpp` revision and could not be applied verbatim to current upstream.

The local port preserves current upstream changes, particularly:

* current `llama_context` rope-scaling initialization;
* current `tests/test-alloc.cpp` dummy backend/device structure;
* current graph-allocation dependency tests.

The adapted `test-alloc` keeps both:

```text
test_shared_buffers
test_graph_optimize_alloc_dep
```

### Important behavior

If a scheduler sharing a compute arena later requires a larger reservation, it detaches and allocates independent storage rather than resizing shared storage that is still referenced by another scheduler.

Expected diagnostic during the allocator test:

```text
ggml_gallocr_reserve_n_impl:
detaching shared compute buffers for a larger reservation
```

This is intentional.

---

## 2. MTP asynchronous synchronization fix

Commit:

```text
common : synchronize MTP draft before target compute reuse
```

This is the critical correctness patch for shared compute buffers.

### Problem

`llama_decode()` schedules graph execution asynchronously.

With #27489 enabled, target and MTP schedulers may share the same physical compute workspace.

The target-to-MTP transition was already effectively synchronized because MTP obtains target embeddings through an accessor which synchronizes the target context.

The reverse transition was not synchronized.

The unsafe sequence was:

```text
target graph
    |
    | synchronized before reading target nextn embeddings
    v
MTP / draft llama_decode()
    |
    | asynchronous execution still in flight
    v
MTP process() returns
    |
    v
target scheduler reuses the same physical compute buffer
```

This can result in simultaneous target and draft access to the same physical compute arena.

Observed symptom on the affected configuration included deterministic output corruption such as:

```text
//
```

or long runs of `/`.

### Fix

After native MTP catch-up decoding completes:

```cpp
if (!ok) {
    return false;
}

// The target and MTP schedulers may share physical compute buffers.
// llama_decode() is asynchronous, so finish the draft work before
// returning control to the target scheduler.
llama_synchronize(ctx_dft);
```

This directly enforces the contract introduced by #27489:

```text
shared schedulers must not execute concurrently
```

The synchronization is intentionally placed at the MTP-to-target handoff.

It is not inserted:

```text
inside llama_decode()
inside generic graph execution
inside the allocator
between every MTP head
```

because doing so would unnecessarily change global asynchronous execution semantics.

### Regression reproducer

A minimal deterministic reproducer was established with:

```text
prompt tokens = 12560
n_predict     = 2
temperature   = 0
seed          = 1
ignore_eos    = true
cache_prompt  = false
```

With #27489 but without the synchronization:

```text
BAD -> //
```

With the synchronization patch:

```text
GOOD
```

The fix was additionally validated under long-context production-like workloads.

### Performance validation

The synchronization does not disable effective MTP speculation.

A representative validated workload showed approximately:

```text
generation             ~48 t/s
draft acceptance       ~74 %
mean accepted length   ~2.47
acceptance per pos     ~0.83 / ~0.64
```

So the barrier fixes the lifetime race while native MTP remains actively useful.

---

## 3. PR #28873

Upstream PR:

```text
#28873

kv-cache : honor LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
in state_write / state_read_sinfo
```

This patch modifies:

```text
src/llama-kv-cache.cpp
```

Its purpose is to make KV-cache state serialization/deserialization correctly respect:

```text
LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
```

This is useful for workflows involving partial sequence-state handling and checkpoint restoration.

It is independent of the compute-buffer sharing implementation and therefore remains a separate commit.

Keeping it separate also makes it easy to remove once upstream merges an equivalent fix.

---

## 4. PR #27451

Upstream PR:

```text
#27451

server : share checkpoint state and harden prompt cache OOM handling
```

Affected areas include:

```text
common/common.cpp
common/common.h
tools/server/server-task.cpp
```

This patch improves server-side prompt checkpoint handling, including shared checkpoint state and more robust cache behavior around memory pressure.

In the current Qwen/MTP workload, server logs demonstrate active checkpoint creation and prompt-cache state management.

Example behavior:

```text
created context checkpoint ...
saving prompt ...
updating prompt cache ...
```

For hybrid/recurrent/SWA models, some checkpoints may still be unusable for a later prompt.

In that situation the server may report:

```text
forcing full prompt re-processing due to lack of cache data
```

followed by deletion of invalidated checkpoints.

This behavior is not itself a failure. The server safely falls back to complete prompt processing instead of restoring incompatible state.

---

# Validation status

The current patch stack has been validated with:

```text
test-alloc                          PASS
test_shared_buffers                 PASS
test_graph_optimize_alloc_dep       PASS

llama-server build                  PASS

minimal MTP corruption reproducer   PASS
32K native MTP                      PASS
65K native MTP                      PASS
mixed CPU/GPU placement             PASS
backend sampling                    PASS
prompt checkpoint/cache operation   PASS
```

The relevant production-like topology includes:

```text
--parallel 1
--ctx-size 65536
--batch-size 4096
--ubatch-size 128

--n-gpu-layers 66

--override-tensor "blk\.(26|27|28)\.ffn_(up|down|gate)\.=CPU"

--cache-type-k q4_0
--cache-type-v q4_0

--flash-attn on

--spec-type draft-mtp
--spec-draft-n-max 2
--spec-draft-type-k q4_0
--spec-draft-type-v q4_0
```

---

# Branch policy

The fork should keep upstream and patched development separate.

Recommended structure:

```text
master
└── exact or near-exact mirror of upstream/master

qwen-mtp
└── upstream/master
    + #27489 port
    + MTP synchronization fix
    + #28873
    + #27451
```

Do not develop directly on `master`.

`master` should remain useful as a clean upstream comparison point.

---

# Updating upstream/master

First update the clean local `master`:

```bat
git fetch upstream

git switch master
git reset --hard upstream/master

git push origin master --force-with-lease
```

Verify:

```bat
git status --short
git log -3 --oneline --decorate
```

The working tree should be clean.

---

# Rebasing qwen-mtp onto a new upstream

Before rebasing, create a safety tag.

Example:

```bat
git switch qwen-mtp

git tag qwen-mtp-pre-rebase-2026-09-23
git push origin qwen-mtp-pre-rebase-2026-09-23
```

Then:

```bat
git fetch upstream

git switch qwen-mtp
git rebase upstream/master
```

Because the branch is a small linear patch series, Git will replay commits individually:

```text
#27489 port
MTP sync fix
#28873
#27451
```

This is intentional.

---

# Resolving rebase conflicts

Do not blindly use:

```bat
git checkout --ours
git checkout --theirs
```

or equivalent whole-file conflict resolution.

The patches are ports across a moving upstream codebase, so current upstream behavior should generally be preserved and only the semantic part of the downstream patch reapplied.

For each conflict:

```bat
git status --short
```

Inspect:

```bat
git diff --cc -- path\to\file
```

Resolve manually.

Then:

```bat
git diff --check -- path\to\file
git add path\to\file
git rebase --continue
```

Repeat until complete.

If the rebase becomes invalid:

```bat
git rebase --abort
```

The pre-rebase tag remains available as a known-good recovery point.

---

# Special rebase considerations

## #27489

This is the patch most likely to conflict with future upstream changes because it touches allocator and scheduler internals.

Pay particular attention to:

```text
ggml/include/ggml-backend.h
ggml/src/ggml-alloc.c
ggml/src/ggml-backend-impl.h
ggml/src/ggml-backend.cpp
src/llama-context.cpp
src/llama-context.h
tests/test-alloc.cpp
```

Do not assume the original #27489 implementation is still correct after upstream allocator changes.

After any conflict resolution, rerun:

```bat
cmake --build build-llvm --target test-alloc -j
build-llvm\bin\Debug\test-alloc.exe
```

At minimum verify:

```text
test_shared_buffers PASSED
test_graph_optimize_alloc_dep PASSED
```

---

## MTP synchronization fix

The key invariant is:

```text
MTP asynchronous work must finish before the target scheduler is allowed
to reuse a compute arena shared with the MTP scheduler.
```

If upstream restructures `common_speculative_impl_draft_mtp`, do not blindly relocate:

```cpp
llama_synchronize(ctx_dft);
```

Instead identify the new MTP-to-target handoff.

The synchronization belongs:

```text
after the final draft/MTP decode
before control returns to target execution
```

If upstream changes `llama_decode()` to synchronous execution or implements an explicit scheduler-level ownership/fence mechanism, reassess whether this patch remains necessary.

---

## #28873

Check whether upstream already implements equivalent handling for:

```text
LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
```

If yes, drop the downstream commit rather than resolving it redundantly.

During a rebase this can be done with:

```bat
git rebase --skip
```

but only after verifying that upstream truly contains equivalent behavior.

---

## #27451

Prompt-cache and checkpoint code changes frequently.

If upstream has substantially rewritten:

```text
common/common.cpp
common/common.h
tools/server/server-task.cpp
```

verify the behavior rather than mechanically reproducing the old diff.

Important properties to retain are:

```text
safe checkpoint state ownership
safe prompt-cache fallback
safe behavior under memory pressure
```

---

# Post-rebase validation

After every upstream rebase, build:

```bat
cmake --build build-llvm --target test-alloc llama-server -j
```

Run:

```bat
build-llvm\bin\Debug\test-alloc.exe
```

Then run the minimal MTP regression case:

```text
prompt tokens = 12560
n_predict     = 2
temperature   = 0
seed          = 1
ignore_eos    = true
cache_prompt  = false
```

Expected:

```text
normal output
no repeated slash corruption
```

Then run one production-like 65K workload.

Check:

```text
no corruption
no CUDA abort
MTP drafts generated
MTP drafts accepted
reasonable generation speed
prompt checkpoints remain functional
```

Only after these pass should the rebased branch replace the previous stable version.

---

# Publishing a rebased qwen-mtp

A rebase rewrites downstream commit hashes.

After successful validation:

```bat
git push origin qwen-mtp --force-with-lease
```

Use:

```text
--force-with-lease
```

rather than plain:

```text
--force
```

because it protects against accidentally overwriting remote changes that were not present locally.

---

# Stable tags

Before experiments or large rebases, create a known-good tag.

Example:

```bat
git tag qwen-mtp-stable-2026-09-23
git push origin qwen-mtp-stable-2026-09-23
```

To return to it temporarily:

```bat
git switch --detach qwen-mtp-stable-2026-09-23
```

To restore the branch to that exact state:

```bat
git switch qwen-mtp
git reset --hard qwen-mtp-stable-2026-09-23
```

Do this only when intentionally discarding newer local branch changes.

---

# Checking whether downstream patches are still needed

Before every major rebase, inspect the relevant upstream PRs.

If an upstream PR has been merged, determine whether the corresponding downstream commit should be removed.

Do not keep both an upstream implementation and an old cherry-picked version unless there is a verified semantic difference.

A useful comparison is:

```bat
git log --oneline upstream/master..qwen-mtp
```

and:

```bat
git diff upstream/master..qwen-mtp
```

The ideal long-term state is a progressively smaller downstream stack as fixes land upstream.

---

# Recovery

Show the known-good history:

```bat
git log --oneline --decorate --graph --all
```

If an experimental update fails:

```bat
git reset --hard qwen-mtp-stable-YYYY-MM-DD
```

or create a new branch from the stable tag:

```bat
git switch -c qwen-mtp-recovery qwen-mtp-stable-YYYY-MM-DD
```

Avoid deleting the stable tag until the replacement branch has passed the full validation suite.

---

# Maintenance principle

Treat `qwen-mtp` as a reproducible patch stack, not as an independent fork of llama.cpp.

The desired lifecycle is:

```text
update upstream
      ↓
rebase patch stack
      ↓
resolve semantic conflicts
      ↓
build
      ↓
test allocator
      ↓
run minimal MTP regression
      ↓
run 65K production validation
      ↓
push with --force-with-lease
      ↓
create/update stable tag
```

This keeps the fork close to upstream while retaining only the changes required for the validated Qwen native-MTP configuration.
