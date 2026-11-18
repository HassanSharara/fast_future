//! Multi-threaded work-stealing runtime for `Send` futures.

use std::cell::Cell;
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering::*};
use std::sync::Mutex;
use std::thread::{self, JoinHandle as ThreadHandle};

use async_task::Runnable;
use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use crossbeam_utils::sync::{Parker, Unparker};
use crossbeam_utils::CachePadded;

use crate::JoinHandle;

/// Every N ticks a worker checks the global queue first (prevents starvation).
const GLOBAL_POLL_INTERVAL: u32 = 61;
/// When a worker's local queue holds this many tasks, new spawns overflow to the
/// global queue so every worker can drain them in parallel.
const LOCAL_QUEUE_MAX: usize = 256;
const SPIN_ROUNDS: u32 = 64;
const YIELD_ROUNDS: u32 = 72;

struct Shared {
    injector: Injector<Runnable>,
    stealers: Vec<Stealer<Runnable>>,
    unparkers: Vec<Unparker>,
    sleepers: Mutex<Vec<usize>>,
    sleeping: CachePadded<AtomicUsize>,
    /// Workers currently trying to steal. Capped at half the pool.
    searching: CachePadded<AtomicUsize>,
    shutdown: AtomicBool,
}

struct WorkerCtx {
    shared: *const Shared,
    local: Worker<Runnable>,
}

thread_local! {
    /// Set on worker threads: lets `schedule` push to the local queue (no contention).
    static CTX: Cell<*const WorkerCtx> = const { Cell::new(ptr::null()) };
    /// Runtime used by the free function `spawn`.
    static CURRENT: Cell<Option<&'static Shared>> = const { Cell::new(None) };
}

impl Shared {
    /// Wake one sleeping worker, if any. Pairs with the fence in `sleep`.
    #[inline]
    fn notify_one(&self) {
        fence(SeqCst);
        // No sleepers, or someone is already searching (it will find the task, or
        // re-check the queues in `sleep` before parking): don't pay for a wakeup.
        if self.sleeping.load(SeqCst) == 0 || self.searching.load(SeqCst) != 0 {
            return;
        }
        let id = {
            let mut s = self.sleepers.lock().unwrap();
            let id = s.pop();
            if id.is_some() {
                self.sleeping.fetch_sub(1, SeqCst);
            }
            id
        };
        if let Some(id) = id {
            self.unparkers[id].unpark();
        }
    }

    fn sleep(&self, local: &Worker<Runnable>, id: usize, parker: &Parker) {
        {
            let mut s = self.sleepers.lock().unwrap();
            s.push(id);
            self.sleeping.fetch_add(1, SeqCst);
        }
        fence(SeqCst);
        // Re-check after registering so a concurrent push can't be missed.
        let has_work = self.shutdown.load(SeqCst)
            || !self.injector.is_empty()
            || !local.is_empty()
            || self.stealers.iter().any(|s| !s.is_empty());
        if !has_work {
            parker.park();
        }
        let mut s = self.sleepers.lock().unwrap();
        if let Some(pos) = s.iter().position(|&x| x == id) {
            s.swap_remove(pos);
            self.sleeping.fetch_sub(1, SeqCst);
        }
    }

    #[inline]
    fn after_steal(&self, local: &Worker<Runnable>) {
        // We grabbed a batch: let another worker help with the rest.
        if !local.is_empty() {
            self.notify_one();
        }
    }

    /// Allow at most half of the workers to hunt for work at once.
    #[inline]
    fn try_start_searching(&self) -> bool {
        if 2 * self.searching.load(Relaxed) >= self.stealers.len() {
            return false;
        }
        self.searching.fetch_add(1, SeqCst);
        true
    }

    /// If the last searcher found work, wake another worker to keep the chain going.
    #[inline]
    fn end_searching(&self, found: bool) {
        let prev = self.searching.fetch_sub(1, SeqCst);
        if found && prev == 1 {
            self.notify_one();
        }
    }

    fn find_task(
        &self,
        local: &Worker<Runnable>,
        id: usize,
        tick: u32,
        rng: &mut u32,
        searching: &mut bool,
    ) -> Option<Runnable> {
        if tick % GLOBAL_POLL_INTERVAL == 0 {
            if let Steal::Success(t) = self.injector.steal_batch_and_pop(local) {
                self.after_steal(local);
                return Some(t);
            }
        }
        if let Some(t) = local.pop() {
            return Some(t);
        }
        loop {
            match self.injector.steal_batch_and_pop(local) {
                Steal::Success(t) => {
                    self.after_steal(local);
                    return Some(t);
                }
                Steal::Empty => break,
                Steal::Retry => {}
            }
        }
        let n = self.stealers.len();
        if n > 1 {
            // Stay in the "searching" state for the whole idle spin (the worker loop
            // ends it), so the shared counter isn't toggled on every failed scan.
            if !*searching {
                if !self.try_start_searching() {
                    return None;
                }
                *searching = true;
            }
            let start = next_rand(rng) as usize % n;
            for i in 0..n {
                let idx = (start + i) % n;
                if idx == id {
                    continue;
                }
                loop {
                    match self.stealers[idx].steal_batch_and_pop(local) {
                        Steal::Success(t) => return Some(t),
                        Steal::Empty => break,
                        Steal::Retry => {}
                    }
                }
            }
        }
        None
    }
}

#[inline]
fn next_rand(s: &mut u32) -> u32 {
    // xorshift32
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

#[inline]
fn schedule(shared: &'static Shared, r: Runnable) {
    let p = CTX.with(|c| c.get());
    if !p.is_null() {
        // SAFETY: CTX is only set on a worker thread for the lifetime of its ctx.
        let ctx = unsafe { &*p };
        if ptr::eq(ctx.shared, shared) {
            if ctx.local.len() < LOCAL_QUEUE_MAX {
                ctx.local.push(r);
            } else {
                shared.injector.push(r);
            }
            shared.notify_one();
            return;
        }
    }
    shared.injector.push(r);
    shared.notify_one();
}

fn spawn_on<F>(shared: &'static Shared, fut: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
{
    let (runnable, task) = async_task::spawn(fut, move |r| schedule(shared, r));
    runnable.schedule();
    task
}


/// main entry point for each thread worker
fn worker_main(shared: &'static Shared, id: usize, local: Worker<Runnable>, parker: Parker) {
    let ctx = WorkerCtx { shared: shared as *const Shared, local };
    CTX.with(|c| c.set(&ctx));
    CURRENT.with(|c| c.set(Some(shared)));

    let mut rng = (id as u32 + 1).wrapping_mul(0x9E37_79B9) | 1;
    let mut tick: u32 = 0;
    let mut idle: u32 = 0;
    let mut searching = false;

    while !shared.shutdown.load(Relaxed) {
        tick = tick.wrapping_add(1);
        match shared.find_task(&ctx.local, id, tick, &mut rng, &mut searching) {
            Some(r) => {
                idle = 0;
                if searching {
                    searching = false;
                    shared.end_searching(true);
                }
                // A panicking task must not kill the worker.
                let _ = catch_unwind(AssertUnwindSafe(|| {
                    r.run();
                }));
            }
            None => {
                idle += 1;
                if idle < SPIN_ROUNDS {
                    std::hint::spin_loop();
                } else if idle < YIELD_ROUNDS {
                    thread::yield_now();
                } else {
                    if searching {
                        searching = false;
                        shared.end_searching(false);
                    }
                    shared.sleep(&ctx.local, id, &parker);
                    idle = 0;
                }
            }
        }
    }

    // Detach ctx first so tasks dropped below reschedule to the injector.
    CTX.with(|c| c.set(ptr::null()));
    while let Some(r) = ctx.local.pop() {
        drop(r);
    }
    CURRENT.with(|c| c.set(None));
}

/// Cloneable handle for spawning onto a [`Runtime`].
#[derive(Clone, Copy)]
pub struct Handle {
    shared: &'static Shared,
}

impl Handle {
    /// Handle of the runtime the current thread belongs to (worker thread or inside `block_on`).
    pub fn current() -> Handle {
        let shared = CURRENT
            .with(|c| c.get())
            .expect("not inside a fast_future::Runtime");
        Handle { shared }
    }

    pub fn spawn<F>(&self, fut: F) -> JoinHandle<F::Output>
        where
            F: Future + Send + 'static,
            F::Output: Send + 'static,
    {
        spawn_on(self.shared, fut)
    }
}

/// Spawn onto the current runtime. Panics outside a runtime context.
pub fn spawn<F>(fut: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
{
    let shared = CURRENT
        .with(|c| c.get())
        .expect("fast_future::spawn called outside a runtime");
    spawn_on(shared, fut)
}

/// Work-stealing multi-threaded runtime.
///
/// The shared scheduler state is intentionally leaked (`&'static`): tasks then
/// need no reference counting on it, which removes a cache line that every
/// spawn and every completion would otherwise fight over.
pub struct Runtime {
    shared: &'static Shared,
    threads: Vec<ThreadHandle<()>>,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Runtime {
    /// `num_threads == 0` means "number of available CPUs".
    pub fn new(num_threads: usize) -> Self {
        let n = if num_threads == 0 {
            thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else {
            num_threads
        };

        let workers: Vec<Worker<Runnable>> = (0..n).map(|_| Worker::new_fifo()).collect();
        let stealers = workers.iter().map(|w| w.stealer()).collect();
        let parkers: Vec<Parker> = (0..n).map(|_| Parker::new()).collect();
        let unparkers = parkers.iter().map(|p| p.unparker().clone()).collect();

        let shared: &'static Shared = Box::leak(Box::new(Shared {
            injector: Injector::new(),
            stealers,
            unparkers,
            sleepers: Mutex::new(Vec::with_capacity(n)),
            sleeping: CachePadded::new(AtomicUsize::new(0)),
            searching: CachePadded::new(AtomicUsize::new(0)),
            shutdown: AtomicBool::new(false),
        }));

        let threads = workers
            .into_iter()
            .zip(parkers)
            .enumerate()
            .map(|(id, (local, parker))| {
                thread::Builder::new()
                    .name(format!("fast_future-worker-{id}"))
                    .spawn(move || worker_main(shared, id, local, parker))
                    .expect("failed to spawn worker thread")
            })
            .collect();

        Runtime { shared, threads }
    }

    pub fn handle(&self) -> Handle {
        Handle { shared: self.shared }
    }

    pub fn spawn<F>(&self, fut: F) -> JoinHandle<F::Output>
        where
            F: Future + Send + 'static,
            F::Output: Send + 'static,
    {
        spawn_on(self.shared, fut)
    }

    /// Run a future to completion on the calling thread. Tasks it spawns run on the workers.
    /// The future itself does not need to be `Send` or `'static`.
    pub fn block_on<F: Future>(&self, fut: F) -> F::Output {
        struct Restore(Option<&'static Shared>);
        impl Drop for Restore {
            fn drop(&mut self) {
                CURRENT.with(|c| c.set(self.0));
            }
        }
        let prev = CURRENT.with(|c| c.replace(Some(self.shared)));
        let _restore = Restore(prev);
        crate::block_on(fut)
    }
}

impl Drop for Runtime {
    /// Stops the workers immediately; unfinished tasks are dropped.
    /// Await your JoinHandles first if you need them to finish.
    fn drop(&mut self) {
        self.shared.shutdown.store(true, SeqCst);
        for u in &self.shared.unparkers {
            u.unpark();
        }
        let me = thread::current().id();
        for t in self.threads.drain(..) {
            if t.thread().id() != me {
                let _ = t.join();
            }
        }
        loop {
            match self.shared.injector.steal() {
                Steal::Success(r) => drop(r),
                Steal::Empty => break,
                Steal::Retry => {}
            }
        }
    }
}