use fast_future::{spawn, spawn_local, LocalRuntime, Runtime};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

const N: usize = 1_000_000;

fn main() {
    // ---- Send runtime (work stealing) ----
    let rt = Runtime::default();
    let counter = Arc::new(AtomicUsize::new(0));
    let t = Instant::now();
    rt.block_on(async {
        let hs: Vec<_> = (0..N)
            .map(|_| {
                let c = counter.clone();
                spawn(async move {
                    c.fetch_add(1, Relaxed);
                })
            })
            .collect();
        for h in hs {
            h.await;
        }
    });
    println!("Runtime:      {N} tasks in {:?} (count={})", t.elapsed(), counter.load(Relaxed));
    drop(rt);

    // ---- Local runtime (!Send futures: Rc/Cell) ----
    let lrt = LocalRuntime::new();
    let c = Rc::new(Cell::new(0usize));
    let t = Instant::now();
    lrt.block_on(async {
        let hs: Vec<_> = (0..N)
            .map(|_| {
                let c = c.clone();
                spawn_local(async move {
                    c.set(c.get() + 1);
                })
            })
            .collect();
        for h in hs {
            h.await;
        }
    });
    println!("LocalRuntime: {N} tasks in {:?} (count={})", t.elapsed(), c.get());

    // ---- Thread-per-core ----
    fast_future::run_thread_per_core(4, |i| async move {
        let local = Rc::new(Cell::new(0));
        let l2 = local.clone();
        spawn_local(async move { l2.set(l2.get() + i) }).await;
        println!("core-thread {i} done, local={}", local.get());
    });
}