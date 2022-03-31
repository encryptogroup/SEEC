use futures::{SinkExt, StreamExt};
use std::fmt::Debug;

pub mod in_memory;
pub mod tcp;
pub(crate) mod util;

pub use in_memory::InMemory;
pub use tcp::Tcp;

// TODO I'm not sure how sensible it is to have one struct for reading and writing
//  if this is split into two structs, reading and writing can be done concurrently
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
