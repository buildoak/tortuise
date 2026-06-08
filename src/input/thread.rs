use crossterm::event;
use std::sync::mpsc::{self, Receiver};
use std::time::Instant;

#[derive(Debug)]
pub enum InputMessage {
    Event {
        event: crossterm::event::Event,
        read_at: Instant,
    },
    ReadError(String),
}

pub type InputReceiver = Receiver<InputMessage>;

pub fn spawn_input_thread() -> InputReceiver {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || loop {
        match event::read() {
            Ok(ev) => {
                if tx
                    .send(InputMessage::Event {
                        event: ev,
                        read_at: Instant::now(),
                    })
                    .is_err()
                {
                    break;
                }
            }
            Err(err) => {
                let _ = tx.send(InputMessage::ReadError(err.to_string()));
                break;
            }
        }
    });
    rx
}
