//! fastrt: two small, fast async runtimes.
//!
//! * [`Runtime`]      - multi-threaded, work-stealing, for `Send` futures.
//! * [`LocalRuntime`] - single-threaded, for `!Send` futures (`Rc`, `RefCell`, ...).
//!   Scheduling is a plain `VecDeque`, no atomics on the hot path.
//!   Use [`run_thread_per_core`] to run one per thread (thread-per-core model).
//!
//! Both use `async-task` (one allocation per task, atomic state machine).
//! Neither includes an I/O reactor or timers; wake tasks from your own
//! reactor (mio / io_uring / timer thread) via ordinary `Waker`s.

mod local;
mod mt;

pub use local::{run_thread_per_core, spawn_local, LocalRuntime};
pub use mt::{spawn, Handle, Runtime};

/// Awaitable handle to a spawned task. NOTE: dropping it *cancels* the task;
/// call `.detach()` to let it run in the background.
pub type JoinHandle<T> = async_task::Task<T>;

use std::future::Future;
use std::pin::{pin, Pin};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

/// Minimal park-based `block_on` for the calling thread.
pub fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
        thread::park();
    }
}

/// Yield once so other tasks can run.
pub async fn yield_now() {
    struct Yield(bool);
    impl Future for Yield {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.0 {
                Poll::Ready(())
            } else {
                self.0 = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
    Yield(false).await
}