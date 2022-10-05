use crate::{channel, Receiver, Sender};
use remoc::RemoteSend;

pub fn new_pair<T: RemoteSend>(
    local_buffer: usize,
) -> ((Sender<T>, Receiver<T>), (Sender<T>, Receiver<T>)) {
    let (sender1, receiver1) = channel(local_buffer);
    let (sender2, receiver2) = channel(local_buffer);

    ((sender1, receiver2), (sender2, receiver1))
}
