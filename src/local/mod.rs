//! Single-threaded runtime for `!Send` futures.
//!
//! Hot path = `VecDeque` push/pop with zero synchronization. Wakes coming from
//! *other* threads (e.g. a channel completed elsewhere) go through a small
//! mutex-protected inbox and unpark the owner thread.

use std::cell::{RefCell, UnsafeCell};
use std::collections::VecDeque;
use std::future::Future;
use std::marker::PhantomData;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::pin::pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::thread;

use async_task::Runnable;
use crossbeam_utils::sync::{Parker, Unparker};

use crate::JoinHandle;

/// Max tasks run between checks of the main future / inbox (fairness).
const BUDGET: usize = 128;

#[inline]
fn thread_token() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! { static ID: u64 = NEXT.fetch_add(1, Relaxed); }
    ID.with(|i| *i)
}

struct LocalShared {
    owner: u64,
    /// Only touched by the owner thread (see `schedule`).
    queue: UnsafeCell<VecDeque<Runnable>>,
    inbox: Mutex<Vec<Runnable>>,
    inbox_flag: AtomicBool,
    closed: AtomicBool,
    unparker: Unparker,
}

// SAFETY: `queue` is only accessed when `thread_token() == owner`; everything else is Sync.
unsafe impl Sync for LocalShared {}

impl LocalShared {
    #[inline]
    fn schedule(&self, r: Runnable) {
        if thread_token() == self.owner {
            if self.closed.load(Relaxed) {
                drop(r); // owner thread: safe to drop the !Send future
                return;
            }
            // SAFETY: owner thread, and no reference into `queue` is live across calls out.
            unsafe { (*self.queue.get()).push_back(r) };
        } else {
            let mut inbox = self.inbox.lock().unwrap();
            if self.closed.load(Relaxed) {
                drop(inbox);
                std::mem::forget(r); // can't drop a !Send future here; leak instead
                return;
            }
            inbox.push(r);
            drop(inbox);
            self.inbox_flag.store(true, Release);
            self.unparker.unpark();
        }
    }

    /// SAFETY: owner thread only.
    #[inline]
    unsafe fn pop(&self) -> Option<Runnable> {
        unsafe { (*self.queue.get()).pop_front() }
    }

    /// SAFETY: owner thread only.
    #[inline]
    unsafe fn drain_inbox(&self) {
        if self.inbox_flag.swap(false, Acquire) {
            let mut inbox = self.inbox.lock().unwrap();
            unsafe { (*self.queue.get()).extend(inbox.drain(..)) };
        }
    }

    fn close(&self) {
        let pending = {
            let mut inbox = self.inbox.lock().unwrap();
            self.closed.store(true, Relaxed);
            std::mem::take(&mut *inbox)
        };
        drop(pending);
        // SAFETY: called from the owner thread (LocalRuntime is !Send).
        while let Some(r) = unsafe { self.pop() } {
            drop(r);
        }
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<LocalShared>>> = const { RefCell::new(None) };
}

fn spawn_in<F>(shared: &Arc<LocalShared>, fut: F) -> JoinHandle<F::Output>
    where
        F: Future + 'static,
        F::Output: 'static,
{
    let s = shared.clone();
    // `spawn_local` enforces (by panicking) that the future is only polled/dropped
    // on this thread, so this is sound even if wakers travel to other threads.
    let (runnable, task) = async_task::spawn_local(fut, move |r| s.schedule(r));
    runnable.schedule();
    task
}

/// Spawn a `!Send` task onto the `LocalRuntime` driving the current thread.
/// Panics if called outside `LocalRuntime::block_on`.
pub fn spawn_local<F>(fut: F) -> JoinHandle<F::Output>
    where
        F: Future + 'static,
        F::Output: 'static,
{
    let shared = CURRENT
        .with(|c| c.borrow().clone())
        .expect("spawn_local called outside LocalRuntime::block_on");
    spawn_in(&shared, fut)
}

struct MainWake {
    flag: AtomicBool,
    owner: u64,
    unparker: Unparker,
}

impl Wake for MainWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.flag.store(true, Release);
        if thread_token() != self.owner {
            self.unparker.unpark();
        }
    }
}

/// Thread-bound runtime. `!Send`: create it on the thread that will run it.
pub struct LocalRuntime {
    shared: Arc<LocalShared>,
    parker: Parker,
    _not_send: PhantomData<*const ()>,
}

impl Default for LocalRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalRuntime {
    pub fn new() -> Self {
        let parker = Parker::new();
        let shared = Arc::new(LocalShared {
            owner: thread_token(),
            queue: UnsafeCell::new(VecDeque::with_capacity(1024)),
            inbox: Mutex::new(Vec::new()),
            inbox_flag: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            unparker: parker.unparker().clone(),
        });
        LocalRuntime { shared, parker, _not_send: PhantomData }
    }

    pub fn spawn_local<F>(&self, fut: F) -> JoinHandle<F::Output>
        where
            F: Future + 'static,
            F::Output: 'static,
    {
        spawn_in(&self.shared, fut)
    }

    /// Drive `fut` and every task spawned with `spawn_local` until `fut` completes.
    /// Spawned tasks that are still pending when `fut` returns stay queued for the next call.
    pub fn block_on<F: Future>(&self, fut: F) -> F::Output {
        struct Restore(Option<Arc<LocalShared>>);
        impl Drop for Restore {
            fn drop(&mut self) {
                CURRENT.with(|c| *c.borrow_mut() = self.0.take());
            }
        }
        let prev = CURRENT.with(|c| c.borrow_mut().replace(self.shared.clone()));
        let _restore = Restore(prev);

        let shared = &*self.shared;
        let main = Arc::new(MainWake {
            flag: AtomicBool::new(true),
            owner: shared.owner,
            unparker: self.parker.unparker().clone(),
        });
        let waker = Waker::from(main.clone());
        let mut cx = Context::from_waker(&waker);
        let mut fut = pin!(fut);

        loop {
            if main.flag.swap(false, Acquire) {
                if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                    return v;
                }
            }

            // SAFETY: we are on the owner thread (LocalRuntime is !Send).
            unsafe { shared.drain_inbox() };

            let mut ran = 0;
            while ran < BUDGET {
                match unsafe { shared.pop() } {
                    Some(r) => {
                        let _ = catch_unwind(AssertUnwindSafe(|| {
                            r.run();
                        }));
                        ran += 1;
                    }
                    None => break,
                }
            }

            if ran == 0 && !main.flag.load(Acquire) && !shared.inbox_flag.load(Acquire) {
                self.parker.park();
            }
        }
    }
}

impl Drop for LocalRuntime {
    fn drop(&mut self) {
        self.shared.close();
    }
}

/// Thread-per-core: spawn `threads` OS threads, each with its own `LocalRuntime`
/// running the future produced by `make(thread_index)`. Blocks until all finish.
/// The future is built *inside* the thread, so it may be `!Send`.
pub fn run_thread_per_core<F, Fut>(threads: usize, make: F)
    where
        F: Fn(usize) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + 'static,
{
    let make = Arc::new(make);
    let handles: Vec<_> = (0..threads)
        .map(|i| {
            let make = make.clone();
            thread::Builder::new()
                .name(format!("fast_future-local-{i}"))
                .spawn(move || {
                    let rt = LocalRuntime::new();
                    rt.block_on(make(i));
                })
                .expect("failed to spawn thread")
        })
        .collect();
    for h in handles {
        h.join().expect("local runtime thread panicked");
    }
}