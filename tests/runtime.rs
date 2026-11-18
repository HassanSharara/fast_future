//! Correctness + stress tests for both runtimes.
//!
//!   cargo test --release -- --test-threads=1
//!   $env:fast_future_SOAK_SECS=300; cargo test --release soak -- --ignored --nocapture
//!
//! If your package is not called `fast_future`, change the line below.
use fast_future as fast_future;

use fast_future::{spawn, spawn_local, yield_now, Handle, LocalRuntime, Runtime};
use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::*};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------- helpers

fn scale(n: usize) -> usize {
    if cfg!(miri) { (n / 500).max(2) } else { n }
}

/// Run `f` on its own thread; panic with a clear message if it doesn't finish in time.
/// A lost wakeup shows up as exactly this kind of hang.
fn with_timeout<F: FnOnce() + Send + 'static>(secs: u64, name: &str, f: F) {
    let (tx, rx) = mpsc::channel();
    let h = thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    match rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(()) => h.join().unwrap(),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("HANG in `{name}` after {secs}s (lost wakeup / deadlock?)")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => match h.join() {
            Ok(()) => {}
            Err(e) => std::panic::resume_unwind(e),
        },
    }
}

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
    fn wait(&self, secs: u64) {
        let g = self.flag.lock().unwrap();
        let (_g, res) = self
            .cv
            .wait_timeout_while(g, Duration::from_secs(secs), |done| !*done)
            .unwrap();
        assert!(
            !res.timed_out(),
            "timed out: {} tasks never finished (lost wakeup?)",
            self.remaining.load(SeqCst)
        );
    }
}

/// A future completed from any thread (exercises cross-thread wakeups).
#[derive(Default)]
struct FlagInner {
    set: bool,
    waker: Option<Waker>,
}
#[derive(Clone, Default)]
struct Flag(Arc<Mutex<FlagInner>>);

impl Flag {
    fn set(&self) {
        let w = {
            let mut g = self.0.lock().unwrap();
            g.set = true;
            g.waker.take()
        };
        if let Some(w) = w {
            w.wake();
        }
    }
}
impl Future for Flag {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let mut g = self.0.lock().unwrap();
        if g.set {
            Poll::Ready(())
        } else {
            g.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

fn spawn_tree(depth: u32, done: Arc<Done>) {
    spawn(async move {
        if depth > 0 {
            spawn_tree(depth - 1, done.clone());
            spawn_tree(depth - 1, done.clone());
        }
        done.finish_one();
    })
        .detach();
}

fn xorshift(s: &mut u32) -> u32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    *s
}

// ------------------------------------------------- multi-threaded Runtime

#[test]
fn spawn_and_join_returns_value() {
    let rt = Runtime::new(4);
    let v = rt.block_on(async { spawn(async { 21 * 2 }).await });
    assert_eq!(v, 42);
}

#[test]
fn many_tasks_all_thread_counts() {
    for threads in [1, 2, 3, 8, 16] {
        with_timeout(60, "many_tasks", move || {
            let rt = Runtime::new(threads);
            let n = scale(200_000);
            let done = Done::new(n);
            for _ in 0..n {
                let d = done.clone();
                rt.spawn(async move { d.finish_one() }).detach();
            }
            done.wait(30);
        });
    }
}

#[test]
fn nested_spawn_tree() {
    for threads in [1, 4, 16] {
        with_timeout(60, "nested_tree", move || {
            let rt = Runtime::new(threads);
            let depth = if cfg!(miri) { 4 } else { 14 };
            let done = Done::new((1usize << (depth + 1)) - 1);
            let d = done.clone();
            rt.spawn(async move { spawn_tree(depth, d) }).detach();
            done.wait(30);
        });
    }
}

#[test]
fn yield_storm_completes() {
    for threads in [1, 8] {
        with_timeout(60, "yield_storm", move || {
            let rt = Runtime::new(threads);
            let tasks = scale(100);
            let done = Done::new(tasks);
            for _ in 0..tasks {
                let d = done.clone();
                rt.spawn(async move {
                    for _ in 0..scale(1000) {
                        yield_now().await;
                    }
                    d.finish_one();
                })
                    .detach();
            }
            done.wait(30);
        });
    }
}

#[test]
fn yielding_task_does_not_starve_others_on_one_thread() {
    with_timeout(20, "starvation", || {
        let rt = Runtime::new(1);
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        rt.spawn(async move {
            while !s.load(Acquire) {
                yield_now().await;
            }
        })
            .detach();
        let done = Done::new(1);
        let (d, s) = (done.clone(), stop.clone());
        rt.spawn(async move {
            s.store(true, Release);
            d.finish_one();
        })
            .detach();
        done.wait(10);
    });
}

#[test]
fn panicking_task_does_not_kill_worker() {
    with_timeout(20, "panic_isolation", || {
        let rt = Runtime::new(1); // only one worker: if it died, nothing could run
        rt.spawn(async {
            if true {
                panic!("expected panic from test task");
            }
        })
            .detach();
        let v = rt.block_on(async { spawn(async { 7 }).await });
        assert_eq!(v, 7);
    });
}

struct DropFlag(Arc<AtomicBool>);
impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, SeqCst);
    }
}

#[test]
fn dropping_join_handle_cancels_and_drops_future() {
    with_timeout(20, "cancel", || {
        let rt = Runtime::new(2);
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = DropFlag(dropped.clone());
        let h = rt.spawn(async move {
            let _g = guard;
            std::future::pending::<()>().await;
        });
        thread::sleep(Duration::from_millis(50));
        drop(h);
        let start = Instant::now();
        while !dropped.load(SeqCst) {
            assert!(start.elapsed() < Duration::from_secs(5), "future never dropped");
            thread::yield_now();
        }
    });
}

#[test]
fn external_threads_wake_tasks() {
    for threads in [1, 4, 16] {
        with_timeout(30, "external_wake", move || {
            let rt = Runtime::new(threads);
            let n = scale(2000);
            let flags: Vec<Flag> = (0..n).map(|_| Flag::default()).collect();
            let done = Done::new(n);
            for f in &flags {
                let (f, d) = (f.clone(), done.clone());
                rt.spawn(async move {
                    f.await;
                    d.finish_one();
                })
                    .detach();
            }
            let setters: Vec<_> = (0..4)
                .map(|t| {
                    let flags = flags.clone();
                    thread::spawn(move || {
                        for (i, f) in flags.iter().enumerate() {
                            if i % 4 == t {
                                f.set();
                                if i % 64 == 0 {
                                    thread::yield_now();
                                }
                            }
                        }
                    })
                })
                .collect();
            for s in setters {
                s.join().unwrap();
            }
            done.wait(10);
        });
    }
}

/// THE important one: bursts followed by idle gaps so every worker falls asleep,
/// then work arrives again. A lost wakeup would hang here.
#[test]
fn burst_idle_burst_no_lost_wakeups() {
    for threads in [2, 4, 16] {
        with_timeout(180, "burst_idle_burst", move || {
            let rt = Runtime::new(threads);
            let mut seed = 0x1234_5678u32 ^ threads as u32;
            for round in 0..scale(2000) {
                let r = xorshift(&mut seed);
                if round % 2 == 0 {
                    // flat burst from outside the runtime
                    let k = 1 + (r % 64) as usize;
                    let done = Done::new(k);
                    for _ in 0..k {
                        let d = done.clone();
                        rt.spawn(async move { d.finish_one() }).detach();
                    }
                    done.wait(10);
                } else {
                    // burst that fans out from inside a worker (local push + steal + chain wake)
                    let depth = 1 + (r % 6);
                    let done = Done::new((1usize << (depth + 1)) - 1);
                    let d = done.clone();
                    rt.spawn(async move { spawn_tree(depth, d) }).detach();
                    done.wait(10);
                }
                // idle gap: mostly short spins, sometimes a real sleep so workers park
                if r % 8 == 0 {
                    thread::sleep(Duration::from_millis(1));
                } else {
                    for _ in 0..(r >> 8) % 3000 {
                        std::hint::spin_loop();
                    }
                }
            }
        });
    }
}

#[test]
fn cross_runtime_spawn_and_handle_current() {
    with_timeout(20, "cross_runtime", || {
        let a = Runtime::new(2);
        let b = Runtime::new(2);
        let hb = b.handle();
        let v = a.block_on(async move {
            let inner = hb.spawn(async { 1 }).await;
            let cur = Handle::current().spawn(async { 2 }).await;
            inner + cur
        });
        assert_eq!(v, 3);
    });
}

#[test]
fn create_and_drop_runtimes_repeatedly() {
    with_timeout(60, "create_drop", || {
        for i in 0..scale(200) {
            let rt = Runtime::new(1 + i % 8);
            for _ in 0..100 {
                rt.spawn(async { yield_now().await }).detach();
            }
            // dropped here with tasks still in flight: must not hang or crash
        }
    });
}

#[test]
#[should_panic(expected = "outside a runtime")]
fn spawn_outside_runtime_panics() {
    let _ = spawn(async {});
}

// ------------------------------------------------------------ LocalRuntime

#[test]
fn local_basic_with_rc() {
    let rt = LocalRuntime::new();
    let c = Rc::new(Cell::new(0usize));
    let c2 = c.clone();
    let out = rt.block_on(async move {
        let hs: Vec<_> = (0..1000usize)
            .map(|i| {
                let c = c2.clone();
                spawn_local(async move {
                    c.set(c.get() + i);
                    i
                })
            })
            .collect();
        let mut sum = 0;
        for h in hs {
            sum += h.await;
        }
        sum
    });
    assert_eq!(out, 499_500);
    assert_eq!(c.get(), 499_500);
}

#[test]
fn local_tasks_run_in_fifo_order() {
    let rt = LocalRuntime::new();
    let log = Rc::new(RefCell::new(Vec::new()));
    let l = log.clone();
    rt.block_on(async move {
        for i in 0..10 {
            let l = l.clone();
            spawn_local(async move { l.borrow_mut().push(i) }).detach();
        }
        yield_now().await;
    });
    assert_eq!(*log.borrow(), (0..10).collect::<Vec<_>>());
}

#[test]
fn local_runtime_woken_from_other_threads() {
    with_timeout(30, "local_cross_thread_wake", || {
        let rt = LocalRuntime::new();
        let n = scale(1000);
        let flags: Vec<Flag> = (0..n).map(|_| Flag::default()).collect();
        let f2 = flags.clone();
        // start waking only after the runtime has parked
        let setter = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            for f in &f2 {
                f.set();
            }
        });
        let count = Rc::new(Cell::new(0usize));
        let c = count.clone();
        rt.block_on(async move {
            let hs: Vec<_> = flags
                .iter()
                .map(|f| {
                    let (f, c) = (f.clone(), c.clone());
                    spawn_local(async move {
                        f.await;
                        c.set(c.get() + 1);
                    })
                })
                .collect();
            for h in hs {
                h.await;
            }
        });
        assert_eq!(count.get(), n);
        setter.join().unwrap();
    });
}

#[test]
fn local_main_future_woken_from_other_thread() {
    with_timeout(20, "local_main_wake", || {
        let rt = LocalRuntime::new();
        let f = Flag::default();
        let f2 = f.clone();
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            f2.set();
        });
        rt.block_on(f);
        t.join().unwrap();
    });
}

#[test]
fn local_runtime_drop_with_pending_tasks() {
    with_timeout(20, "local_drop", || {
        for _ in 0..scale(100) {
            let rt = LocalRuntime::new();
            let c = Rc::new(Cell::new(0));
            let c2 = c.clone();
            rt.block_on(async move {
                for _ in 0..50 {
                    let c = c2.clone();
                    spawn_local(async move {
                        yield_now().await;
                        c.set(c.get() + 1);
                    })
                        .detach();
                }
            });
            // dropped with queued !Send tasks: must drop them on this thread without panicking
        }
    });
}

#[test]
#[should_panic(expected = "outside LocalRuntime")]
fn spawn_local_outside_runtime_panics() {
    let _ = spawn_local(async {});
}

static CORE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[test]
fn thread_per_core_runs_every_thread() {
    with_timeout(30, "thread_per_core", || {
        fast_future::run_thread_per_core(4, |i| async move {
            let local = Rc::new(Cell::new(0));
            let l = local.clone();
            spawn_local(async move { l.set(i + 1) }).await;
            assert_eq!(local.get(), i + 1);
            CORE_COUNT.fetch_add(1, SeqCst);
        });
        assert_eq!(CORE_COUNT.load(SeqCst), 4);
    });
}

// -------------------------------------------------------------------- soak

/// Long random soak. Ignored by default:
///   $env:fast_future_SOAK_SECS=600; cargo test --release soak -- --ignored --nocapture
#[test]
#[ignore]
fn soak() {
    let secs: u64 = std::env::var("fast_future_SOAK_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    with_timeout(secs + 120, "soak", move || {
        let start = Instant::now();
        let mut seed = 0xDEAD_BEEFu32;
        let mut rounds = 0u64;
        while start.elapsed() < Duration::from_secs(secs) {
            // new runtime with a random thread count every few seconds
            let threads = 1 + (xorshift(&mut seed) % 16) as usize;
            let rt = Runtime::new(threads);
            let epoch = Instant::now();
            while epoch.elapsed() < Duration::from_secs(3) && start.elapsed() < Duration::from_secs(secs) {
                let r = xorshift(&mut seed);
                match r % 3 {
                    0 => {
                        let k = 1 + (r >> 4) as usize % 500;
                        let done = Done::new(k);
                        for _ in 0..k {
                            let d = done.clone();
                            rt.spawn(async move {
                                if r % 5 == 0 {
                                    yield_now().await;
                                }
                                d.finish_one();
                            })
                                .detach();
                        }
                        done.wait(15);
                    }
                    1 => {
                        let depth = 1 + (r >> 4) % 10;
                        let done = Done::new((1usize << (depth + 1)) - 1);
                        let d = done.clone();
                        rt.spawn(async move { spawn_tree(depth, d) }).detach();
                        done.wait(15);
                    }
                    _ => {
                        let n = 1 + (r >> 4) as usize % 200;
                        let flags: Vec<Flag> = (0..n).map(|_| Flag::default()).collect();
                        let done = Done::new(n);
                        for f in &flags {
                            let (f, d) = (f.clone(), done.clone());
                            rt.spawn(async move {
                                f.await;
                                d.finish_one();
                            })
                                .detach();
                        }
                        let t = thread::spawn(move || flags.iter().for_each(|f| f.set()));
                        done.wait(15);
                        t.join().unwrap();
                    }
                }
                rounds += 1;
                if r % 16 == 0 {
                    thread::sleep(Duration::from_millis(2));
                }
            }
        }
        println!("soak finished: {rounds} rounds, no hangs");
    });
}