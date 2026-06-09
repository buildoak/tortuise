use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct Slot<T> {
    value: Option<T>,
    published: u64,
    superseded: u64,
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self {
            value: None,
            published: 0,
            superseded: 0,
        }
    }
}

#[derive(Debug)]
pub struct LatestDrain<T> {
    pub value: Option<T>,
    pub messages: usize,
    pub dropped_or_superseded: u64,
}

#[derive(Debug)]
pub struct LatestBus<T> {
    slot: Arc<Mutex<Slot<T>>>,
}

impl<T> LatestBus<T> {
    pub fn new() -> Self {
        Self {
            slot: Arc::new(Mutex::new(Slot::default())),
        }
    }

    pub fn publish(&self, value: T) {
        let mut slot = self.slot.lock().expect("latest bus mutex poisoned");
        if slot.value.is_some() {
            slot.superseded += 1;
        }
        slot.value = Some(value);
        slot.published += 1;
    }

    pub fn take_latest(&self) -> LatestDrain<T> {
        let mut slot = self.slot.lock().expect("latest bus mutex poisoned");
        let value = slot.value.take();
        let messages = usize::from(value.is_some());
        let dropped_or_superseded = slot.superseded;
        slot.superseded = 0;
        LatestDrain {
            value,
            messages,
            dropped_or_superseded,
        }
    }
}

impl<T> Clone for LatestBus<T> {
    fn clone(&self) -> Self {
        Self {
            slot: Arc::clone(&self.slot),
        }
    }
}

impl<T> Default for LatestBus<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newest_sample_wins_and_backlog_is_bounded() {
        let bus = LatestBus::new();
        bus.publish(1);
        bus.publish(2);
        bus.publish(3);

        let drain = bus.take_latest();
        assert_eq!(drain.value, Some(3));
        assert_eq!(drain.messages, 1);
        assert_eq!(drain.dropped_or_superseded, 2);

        let empty = bus.take_latest();
        assert_eq!(empty.value, None);
        assert_eq!(empty.messages, 0);
        assert_eq!(empty.dropped_or_superseded, 0);
    }
}
