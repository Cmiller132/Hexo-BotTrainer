//! Background deep-solve pool (Stage 4 async rung, PLAN_TSS_DEEPENING.md §10).
//!
//! Moves the deep leaf solves off the search's critical path: gated leaves
//! ENQUEUE a solve request and proceed to the normal GPU eval; pool workers
//! run the identical verified path (`tree::tss_solve_verified` — solver →
//! independent certificate verifier → sealed `HardValue` mint) on their own
//! threads; the driver drains completed results back into the owning search's
//! per-move memo, where the descent-stop in `select_pending_leaf` consumes
//! them on every later visit through the proven position.
//!
//! Soundness is inherited wholesale: nothing here can mint a hard value —
//! only route one that `tss_core::hard_value_from_verified` already accepted.
//! What the pool DOES change is timing: which visit first sees a proof is
//! wall-clock dependent, so flag-on self-play is NOT bit-reproducible under a
//! fixed seed (the flag-off golden digest remains the bit-identity anchor).
//!
//! Staleness: every request carries the pool-global GENERATION its search
//! held at enqueue time (re-assigned on every move/rebind). A response whose
//! generation no longer matches the slot's live search is dropped — except
//! its fatal `deep_verify_failed` count, which is never dropped.
//!
//! Memory: the request queue is bounded (`try_send`; a full queue drops the
//! request and counts it — search never blocks), each worker's solver TT is
//! byte-capped per solve exactly like the inline path, and responses carry
//! only scalars + the small `RootBinding`.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, Sender, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use hexo_engine::HexoState as RustHexoState;
use hexo_utils::StateHash;

use crate::tree::{tss_solve_verified, TssCounters};
use crate::tss_core::{HardValue, ProofStatus, SolveGoal};
use crate::tss_solver::TssSolver;
use crate::tss_verify::RootBinding;

/// Bounded request-queue depth. Full queue => the leaf silently falls back to
/// the plain net eval (counted as `async_dropped`) — backpressure must never
/// stall selection. Sized for main_3's worst-case burst geometry (256 slots ×
/// 96-leaf batches in threat-dense passes); memory cost is one state clone
/// per entry (~KBs each).
pub const TSS_ASYNC_QUEUE_CAP: usize = 16384;

/// Out-of-band alarm channel, written by WORKERS at solve time so the fatal
/// signal exists the moment it happens — it can never be lost to a dropped,
/// stale, or never-drained response (Codex review 4). The drain passes fold
/// pending failures into a live search's counters (=> epoch telemetry); an
/// untaken residue is screamed about on pool drop.
#[derive(Default)]
pub struct PoolAlarms {
    pub verify_failed: AtomicU32,
    pub worker_panics: AtomicU32,
}

/// A gated leaf's solve request. `state` is a clone taken at enqueue time;
/// `binding` re-asserts full-position identity on the way back (the 64-bit
/// hash is never trusted alone for a value-bearing result, §2.5).
pub struct SolveRequest {
    pub slot: u32,
    pub generation: u64,
    pub hash: StateHash,
    pub binding: RootBinding,
    pub state: RustHexoState,
    pub node_cap: u64,
    pub goal: SolveGoal,
}

/// A completed, already-verified solve. `hard` is `Some` only when the
/// independent verifier accepted the certificate inside `tss_solve_verified`
/// on the worker thread; `counters` carries that solve's telemetry deltas
/// (deep_calls/win/loss/unknown/nodes/verify_failed) for the owning move.
pub struct SolveResponse {
    pub slot: u32,
    pub generation: u64,
    pub hash: StateHash,
    pub binding: RootBinding,
    pub status: ProofStatus,
    pub hard: Option<HardValue>,
    pub counters: TssCounters,
}

/// Per-search enqueue handle: a clone of the pool's bounded sender plus the
/// slot/generation identity stamped on every request. Rewired by the driver
/// at every search creation, reuse-rebind, and move advance.
#[derive(Clone)]
pub struct TssAsyncHandle {
    pub sender: SyncSender<SolveRequest>,
    pub slot: u32,
    pub generation: u64,
    queue_depth: Arc<AtomicUsize>,
}

impl std::fmt::Debug for TssAsyncHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TssAsyncHandle")
            .field("slot", &self.slot)
            .field("generation", &self.generation)
            .finish()
    }
}

impl TssAsyncHandle {
    /// Cheap saturation pre-check (approximate — the depth counter is
    /// relaxed): callers skip the state clone + request build entirely when
    /// the queue is full instead of paying for a doomed `try_send`
    /// (Codex review 5).
    pub fn has_capacity(&self) -> bool {
        self.queue_depth.load(Ordering::Relaxed) < TSS_ASYNC_QUEUE_CAP
    }

    /// Non-blocking enqueue. `false` => queue full or pool gone (caller counts
    /// `async_dropped` and the leaf takes the plain net eval).
    pub fn try_enqueue(&self, request: SolveRequest) -> bool {
        // Increment BEFORE the send (compensating on failure): the worker's
        // post-recv decrement can then never race ahead of the producer's
        // increment and wrap the counter below zero. The counter may briefly
        // over-count — the safe direction for a full-check.
        self.queue_depth.fetch_add(1, Ordering::Relaxed);
        match self.sender.try_send(request) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.queue_depth.fetch_sub(1, Ordering::Relaxed);
                false
            }
        }
    }
}

/// The worker pool. Owned by the MCTS session so worker solvers (each with
/// its own persistent positive-proof-fragment cache) stay warm across
/// `run_continuous` calls. Dropping the pool closes the request channel and
/// the workers exit on their next `recv`.
pub struct TssAsyncPool {
    /// `Some` for the pool's whole life; taken (dropped) in `Drop` to close
    /// the channel so workers exit before the final alarm read.
    request_tx: Option<SyncSender<SolveRequest>>,
    results: Receiver<SolveResponse>,
    generation: AtomicU64,
    queue_depth: Arc<AtomicUsize>,
    alarms: Arc<PoolAlarms>,
    shutdown: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for TssAsyncPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TssAsyncPool")
            .field("workers", &self.workers.len())
            .finish()
    }
}

impl TssAsyncPool {
    pub fn new(threads: u32) -> Self {
        let threads = threads.clamp(1, 32) as usize;
        let (request_tx, request_rx) = sync_channel::<SolveRequest>(TSS_ASYNC_QUEUE_CAP);
        let (response_tx, results) = std::sync::mpsc::channel::<SolveResponse>();
        // std mpsc receivers are single-consumer; workers share it behind a
        // mutex. Lock hold time is one `recv` against multi-microsecond
        // solves, so contention is negligible.
        let request_rx = Arc::new(Mutex::new(request_rx));
        let queue_depth = Arc::new(AtomicUsize::new(0));
        let alarms = Arc::new(PoolAlarms::default());
        let shutdown = Arc::new(AtomicBool::new(false));
        let workers = (0..threads)
            .map(|index| {
                let rx = Arc::clone(&request_rx);
                let tx = response_tx.clone();
                let depth = Arc::clone(&queue_depth);
                let alarms = Arc::clone(&alarms);
                let stop = Arc::clone(&shutdown);
                std::thread::Builder::new()
                    .name(format!("tss-solve-{index}"))
                    .spawn(move || worker_loop(rx, tx, depth, alarms, stop))
                    .expect("spawn tss async solve worker")
            })
            .collect();
        Self {
            request_tx: Some(request_tx),
            results,
            generation: AtomicU64::new(1),
            queue_depth,
            alarms,
            shutdown,
            workers,
        }
    }

    /// Mint a fresh generation (monotone, pool-global — unique per
    /// (search, move) so cross-move and cross-call responses can never
    /// masquerade as live).
    pub fn next_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::Relaxed)
    }

    /// A handle stamped for `slot` at a fresh generation.
    pub fn handle_for(&self, slot: u32) -> TssAsyncHandle {
        TssAsyncHandle {
            sender: self
                .request_tx
                .as_ref()
                .expect("request channel lives until Drop")
                .clone(),
            slot,
            generation: self.next_generation(),
            queue_depth: Arc::clone(&self.queue_depth),
        }
    }

    /// Drain every completed response without blocking.
    pub fn try_drain(&self) -> Vec<SolveResponse> {
        let mut drained = Vec::new();
        while let Ok(response) = self.results.try_recv() {
            drained.push(response);
        }
        drained
    }

    /// Take (swap to 0) the accumulated fatal verify-failure count. Drain
    /// passes call this with a live search in hand so the count reaches the
    /// epoch telemetry no matter which response carried the failure.
    pub fn take_verify_failures(&self) -> u32 {
        self.alarms.verify_failed.swap(0, Ordering::Relaxed)
    }

    /// Take (swap to 0) the accumulated worker-panic count (ops signal; each
    /// panic lost one request and recycled that worker's solver).
    pub fn take_worker_panics(&self) -> u32 {
        self.alarms.worker_panics.swap(0, Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for TssAsyncPool {
    fn drop(&mut self) {
        // Quiesce BEFORE the final alarm read (a worker mid-solve could bank
        // a failure after an early read): raise the shutdown flag (workers
        // exit after at most their CURRENT solve instead of draining the
        // whole buffered queue), close the request channel, then join. Only
        // then is the alarm bank final. Handles held by searches keep their
        // own sender clones, but by pool-drop time the owning session (and
        // its searches) are gone — and even a straggler clone only makes a
        // worker's send fail AFTER its alarms were banked and joined here.
        self.shutdown.store(true, Ordering::Relaxed);
        self.request_tx = None;
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
        let verify_failed = self.alarms.verify_failed.load(Ordering::Relaxed);
        let panics = self.alarms.worker_panics.load(Ordering::Relaxed);
        if verify_failed > 0 {
            eprintln!(
                "hexfield tss_async: {verify_failed} UNREPORTED certificate verify \
                 FAILURE(s) at pool shutdown — investigate immediately"
            );
        }
        if panics > 0 {
            eprintln!("hexfield tss_async: {panics} unreported worker panic(s) at pool shutdown");
        }
    }
}

fn worker_loop(
    rx: Arc<Mutex<Receiver<SolveRequest>>>,
    tx: Sender<SolveResponse>,
    queue_depth: Arc<AtomicUsize>,
    alarms: Arc<PoolAlarms>,
    shutdown: Arc<AtomicBool>,
) {
    // One persistent solver per worker: its shared positive-proof-fragment TT
    // warms across solves (O16); byte caps are enforced per solve inside
    // `tss_solve_verified` exactly as on the inline path.
    let mut solver = TssSolver::default();
    loop {
        if shutdown.load(Ordering::Relaxed) {
            return; // pool dropping: don't drain the buffered queue
        }
        let request = {
            let guard = match rx.lock() {
                Ok(guard) => guard,
                // A poisoned lock means a sibling worker panicked while
                // HOLDING the recv lock (the panic shield below covers the
                // solve, not the recv). Exit; enqueues degrade to net evals.
                Err(_) => return,
            };
            // recv_timeout, NOT recv: a handle clone parked on a search can
            // keep the channel connected past pool drop, and a worker
            // blocked in a plain recv would never recheck `shutdown` — the
            // Drop join would deadlock (Codex round 3). The timeout bounds
            // every worker's reaction to the flag.
            match guard.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(request) => request,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
            }
        };
        queue_depth.fetch_sub(1, Ordering::Relaxed);
        if shutdown.load(Ordering::Relaxed) {
            return; // checked again post-recv: skip the doomed solve
        }
        // Panic shield (Codex review 7): a panicking solve loses its request
        // (the Pending entry falls out at the owner's next move) but the
        // worker survives with a FRESH solver (the old one's state is
        // suspect), and the panic is counted instead of silently shrinking
        // the pool.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut counters = TssCounters::default();
            let solved = tss_solve_verified(
                &request.state,
                request.node_cap,
                request.goal,
                &mut solver,
                &mut counters,
            );
            (solved.status, solved.hard, counters)
        }));
        let (status, hard, mut counters) = match outcome {
            Ok(result) => result,
            Err(_) => {
                alarms.worker_panics.fetch_add(1, Ordering::Relaxed);
                solver = TssSolver::default();
                continue;
            }
        };
        // The alarm atomic is the SOLE carrier of the fatal signal (single
        // channel — no drain-vs-response double count): strip it from the
        // response counters after banking it.
        if counters.deep_verify_failed > 0 {
            alarms
                .verify_failed
                .fetch_add(counters.deep_verify_failed, Ordering::Relaxed);
            counters.deep_verify_failed = 0;
        }
        let response = SolveResponse {
            slot: request.slot,
            generation: request.generation,
            hash: request.hash,
            binding: request.binding,
            status,
            hard,
            counters,
        };
        if tx.send(response).is_err() {
            return; // pool dropped
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::tss_solve_verified;
    use crate::tss_solver::TssSolver;
    use hexo_engine::{apply_placement, HexCoord, Placement};

    fn replay(coords: &[(i16, i16)]) -> RustHexoState {
        let mut state = RustHexoState::new();
        for &(q, r) in coords {
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord { q, r },
                },
            )
            .unwrap();
        }
        state
    }

    /// The tss_solver.rs win-now fixture: the mover completes a win, so the
    /// solve is decided (not Unknown) within a tiny cap.
    fn win_now_fixture() -> RustHexoState {
        replay(&[
            (0, 0),
            (0, 8),
            (2, 7),
            (1, 0),
            (2, 0),
            (4, 6),
            (6, 5),
            (3, 0),
            (4, 0),
            (8, 4),
            (10, 3),
        ])
    }

    fn drain_one(pool: &TssAsyncPool) -> SolveResponse {
        for _ in 0..1000 {
            let mut drained = pool.try_drain();
            if !drained.is_empty() {
                assert_eq!(drained.len(), 1, "exactly one response expected");
                return drained.pop().unwrap();
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("async solve pool produced no response within 10s");
    }

    /// A pool worker must return the identical verified result the inline
    /// path computes, stamped with the request's slot/generation/identity.
    #[test]
    fn pool_round_trip_matches_inline_verified_solve() {
        let state = win_now_fixture();
        let hash: StateHash = 0xDEAD_BEEF; // routing key only; opaque to the pool
        let binding = RootBinding::from_state(&state);

        let mut inline_counters = crate::tree::TssCounters::default();
        let inline = tss_solve_verified(
            &state,
            2000,
            SolveGoal::Both,
            &mut TssSolver::default(),
            &mut inline_counters,
        );
        assert_ne!(inline.status, ProofStatus::Unknown, "fixture must be decided");
        assert!(inline.hard.is_some(), "decided fixture must verify");

        let pool = TssAsyncPool::new(2);
        assert_eq!(pool.worker_count(), 2);
        let handle = pool.handle_for(7);
        assert!(handle.try_enqueue(SolveRequest {
            slot: handle.slot,
            generation: handle.generation,
            hash,
            binding: binding.clone(),
            state: state.clone(),
            node_cap: 2000,
            goal: SolveGoal::Both,
        }));
        let response = drain_one(&pool);
        assert_eq!(response.slot, 7);
        assert_eq!(response.generation, handle.generation);
        assert_eq!(response.hash, hash);
        assert_eq!(response.binding, binding);
        assert_eq!(response.status, inline.status);
        assert_eq!(response.hard.is_some(), inline.hard.is_some());
        assert!(response.counters.deep_calls == 1);
        assert_eq!(response.counters.deep_verify_failed, 0);
    }

    /// Generations are unique and monotone — the staleness guarantee's
    /// foundation (a response minted under an old generation can never
    /// collide with a live one).
    #[test]
    fn generations_are_unique_and_monotone() {
        let pool = TssAsyncPool::new(1);
        let a = pool.handle_for(0).generation;
        let b = pool.handle_for(0).generation;
        let c = pool.handle_for(3).generation;
        assert!(a < b && b < c);
    }
}
