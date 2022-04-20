use futures::{SinkExt, StreamExt};

pub use in_memory::InMemory;
pub use tcp::Tcp;

pub mod in_memory;
pub mod tcp;
pub(crate) mod util;

// TODO I'm not sure how sensible it is to have one struct for reading and writing
//  if this is split into two structs, reading and writing can be done concurrently
pub trait Transport<Item>:
    SinkExt<Item, Error = Self::SinkError> + StreamExt<Item = Result<Item, Self::StreamError>> + Unpin
{
    type SinkError;
    type StreamError;
}

impl<Item, T: Transport<Item>> Transport<Item> for &mut T {
    type SinkError = T::SinkError;
    type StreamError = T::StreamError;
}
