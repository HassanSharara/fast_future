//! fast_future vs tokio.   Run with:
//!   cargo run --release --examples bench            (all CPUs)
//!   cargo run --release --examples bench -- 4       (4 worker threads)
//!
//! Each benchmark does 1 warmup run + ITERS measured runs and reports min / median.

use std::cell::Cell;
use std::future::Future;
use std::hint::black_box;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const ITERS: usize = 7;

// ---------------------------------------------------------------------------
// Runtime-agnostic completion signal (same cost for both runtimes)
// ---------------------------------------------------------------------------
struct Done {
    remaining: AtomicUsize,
    flag: Mutex<bool>,
    cv: Condvar,
}

impl Done {
    fn new(n: usize) -> Arc<Self> {
        Arc::new(Done { remaining: AtomicUsize::new(n), flag: Mutex::new(false), cv: Condvar::new() })
    }
    fn finish_one(&self) {
        if self.remaining.fetch_sub(1, AcqRel) == 1 {
            *self.flag.lock().unwrap() = true;
            self.cv.notify_all();
        }
    }
    fn wait(&self) {
        let mut g = self.flag.lock().unwrap();
        while !*g {
            g = self.cv.wait(g).unwrap();
        }
    }
}

#[inline(never)]
fn burn(iters: u32) {
    let mut x = 1u64;
    for i in 0..iters {
        x = black_box(x.wrapping_mul(6364136223846793005).wrapping_add(i as u64));
    }
    black_box(x);
}

// ---------------------------------------------------------------------------
// Multi-threaded backends
// ---------------------------------------------------------------------------
trait Backend: 'static {
    /// Spawn from outside the runtime (main thread).
    fn spawn_root<F: Future<Output = ()> + Send + 'static>(&self, f: F);
    /// Spawn from inside a running task.
    fn spawn<F: Future<Output = ()> + Send + 'static>(f: F);
    fn yield_now() -> impl Future<Output = ()> + Send;
}

struct Fast(fast_future::Runtime);
impl Backend for Fast {
    fn spawn_root<F: Future<Output = ()> + Send + 'static>(&self, f: F) {
        self.0.spawn(f).detach();
    }
    fn spawn<F: Future<Output = ()> + Send + 'static>(f: F) {
        fast_future::spawn(f).detach();
    }
    fn yield_now() -> impl Future<Output = ()> + Send {
        fast_future::yield_now()
    }
}

struct Tok(tokio::runtime::Runtime);
impl Backend for Tok {
    fn spawn_root<F: Future<Output = ()> + Send + 'static>(&self, f: F) {
        self.0.spawn(f);
    }
    fn spawn<F: Future<Output = ()> + Send + 'static>(f: F) {
        tokio::spawn(f);
    }
    fn yield_now() -> impl Future<Output = ()> + Send {
        tokio::task::yield_now()
    }
}

/// n tiny tasks spawned from the main thread (all go through the global queue).
fn ext_spawn<B: Backend>(b: &B, n: usize) -> Duration {
    let done = Done::new(n);
    let t = Instant::now();
    for _ in 0..n {
        let d = done.clone();
        b.spawn_root(async move { d.finish_one() });
    }
    done.wait();
    t.elapsed()
}

/// One root task spawns n tasks from inside a worker (local queue + stealing).
fn fanout<B: Backend>(b: &B, n: usize, work: u32) -> Duration {
    let done = Done::new(n);
    let t = Instant::now();
    let d = done.clone();
    b.spawn_root(async move {
        for _ in 0..n {
            let d = d.clone();
            B::spawn(async move {
                burn(work);
                d.finish_one();
            });
        }
    });
    done.wait();
    t.elapsed()
}

fn spawn_tree<B: Backend>(depth: u32, done: Arc<Done>) {
    B::spawn(async move {
        if depth > 0 {
            spawn_tree::<B>(depth - 1, done.clone());
            spawn_tree::<B>(depth - 1, done.clone());
        }
        done.finish_one();
    });
}

/// Binary tree of tasks: every task spawns two children. Stresses work stealing.
fn tree<B: Backend>(b: &B, depth: u32) -> Duration {
    let total = (1usize << (depth + 1)) - 1;
    let done = Done::new(total);
    let t = Instant::now();
    let d = done.clone();
    b.spawn_root(async move { spawn_tree::<B>(depth, d) });
    done.wait();
    t.elapsed()
}

/// `tasks` tasks that each yield `yields` times: stresses wake + reschedule.
fn yielding<B: Backend>(b: &B, tasks: usize, yields: usize) -> Duration {
    let done = Done::new(tasks);
    let t = Instant::now();
    for _ in 0..tasks {
        let d = done.clone();
        b.spawn_root(async move {
            for _ in 0..yields {
                B::yield_now().await;
            }
            d.finish_one();
        });
    }
    done.wait();
    t.elapsed()
}

// ---------------------------------------------------------------------------
// Single-threaded (!Send) benchmarks: fast_future::LocalRuntime vs tokio LocalSet
// ---------------------------------------------------------------------------
fn local_spawn_fast(n: usize) -> Duration {
    let rt = fast_future::LocalRuntime::new();
    let c = Rc::new(Cell::new(0usize));
    let t = Instant::now();
    rt.block_on(async {
        for _ in 0..n {
            let c = c.clone();
            fast_future::spawn_local(async move { c.set(c.get() + 1) }).detach();
        }
        while c.get() < n {
            fast_future::yield_now().await;
        }
    });
    t.elapsed()
}

fn local_spawn_tokio(n: usize) -> Duration {
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let local = tokio::task::LocalSet::new();
    let c = Rc::new(Cell::new(0usize));
    let t = Instant::now();
    local.block_on(&rt, async {
        for _ in 0..n {
            let c = c.clone();
            tokio::task::spawn_local(async move { c.set(c.get() + 1) });
        }
        while c.get() < n {
            tokio::task::yield_now().await;
        }
    });
    t.elapsed()
}

fn local_yield_fast(tasks: usize, yields: usize) -> Duration {
    let rt = fast_future::LocalRuntime::new();
    let c = Rc::new(Cell::new(0usize));
    let t = Instant::now();
    rt.block_on(async {
        for _ in 0..tasks {
            let c = c.clone();
            fast_future::spawn_local(async move {
                for _ in 0..yields {
                    fast_future::yield_now().await;
                }
                c.set(c.get() + 1);
            })
                .detach();
        }
        while c.get() < tasks {
            fast_future::yield_now().await;
        }
    });
    t.elapsed()
}

fn local_yield_tokio(tasks: usize, yields: usize) -> Duration {
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let local = tokio::task::LocalSet::new();
    let c = Rc::new(Cell::new(0usize));
    let t = Instant::now();
    local.block_on(&rt, async {
        for _ in 0..tasks {
            let c = c.clone();
            tokio::task::spawn_local(async move {
                for _ in 0..yields {
                    tokio::task::yield_now().await;
                }
                c.set(c.get() + 1);
            });
        }
        while c.get() < tasks {
            tokio::task::yield_now().await;
        }
    });
    t.elapsed()
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
fn run(name: &str, units: usize, mut f: impl FnMut() -> Duration) -> Duration {
    let _ = f(); // warmup
    let mut v: Vec<Duration> = (0..ITERS).map(|_| f()).collect();
    v.sort();
    let (min, med) = (v[0], v[v.len() / 2]);
    println!(
        "  {name:<7} min {min:>10.2?}   median {med:>10.2?}   {:>8.2} M ops/s",
        units as f64 / med.as_secs_f64() / 1e6
    );
    med
}

fn compare(
    title: &str,
    units: usize,
    fast: impl FnMut() -> Duration,
    tok: impl FnMut() -> Duration,
) {
    println!("\n{title}  ({units} ops)");
    let a = run("fast_future", units, fast);
    let b = run("tokio", units, tok);
    let r = b.as_secs_f64() / a.as_secs_f64();
    if r >= 1.0 {
        println!("  => fast_future is {r:.2}x FASTER");
    } else {
        println!("  => fast_future is {:.2}x SLOWER", 1.0 / r);
    }
}

fn main() {
    if cfg!(debug_assertions) {
        eprintln!("WARNING: debug build! Use `cargo run --release --examples bench`.\n");
    }
    let threads = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    println!("worker threads: {threads}, iterations: {ITERS} (+1 warmup)");

    let fast = Fast(fast_future::Runtime::new(threads));
    let tok = Tok(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(threads)
            .build()
            .unwrap(),
    );

    println!("\n=== Multi-threaded (Send) ===");
    const N: usize = 1_000_000;
    compare("spawn from outside runtime", N, || ext_spawn(&fast, N), || ext_spawn(&tok, N));
    compare("fan-out from inside a task", N, || fanout(&fast, N, 0), || fanout(&tok, N, 0));
    compare(
        "fan-out + ~500 iters of work/task",
        N / 2,
        || fanout(&fast, N / 2, 500),
        || fanout(&tok, N / 2, 500),
    );
    compare(
        "binary task tree (depth 19)",
        (1 << 20) - 1,
        || tree(&fast, 19),
        || tree(&tok, 19),
    );
    compare(
        "yield storm (1000 tasks x 1000 yields)",
        1_000_000,
        || yielding(&fast, 1000, 1000),
        || yielding(&tok, 1000, 1000),
    );

    println!("\n=== Single-threaded (!Send): LocalRuntime vs tokio current_thread + LocalSet ===");
    compare("spawn_local + run", N, || local_spawn_fast(N), || local_spawn_tokio(N));
    compare(
        "local yield storm (1000 x 1000)",
        1_000_000,
        || local_yield_fast(1000, 1000),
        || local_yield_tokio(1000, 1000),
    );
}