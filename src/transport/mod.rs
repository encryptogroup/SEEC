use futures::{SinkExt, StreamExt};
use std::fmt::Debug;

pub mod in_memory;
pub mod tcp;
pub(crate) mod util;

pub use in_memory::InMemory;
pub use tcp::Tcp;

pub trait Transport<Item, SinkErr: Debug, StreamErr: Debug>:
    SinkExt<Item, Error = SinkErr> + StreamExt<Item = Result<Item, StreamErr>> + Unpin
{
}
impl<
        Item,
        SinkErr: Debug,
        StreamErr: Debug,
        C: SinkExt<Item, Error = SinkErr> + StreamExt<Item = Result<Item, StreamErr>> + Unpin,
    > Transport<Item, SinkErr, StreamErr> for C
{
}
