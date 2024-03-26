//! Multiplication triple storage of pre-computed MTs.
use crate::mul_triple::MTProvider;
use crate::protocols::SetupStorage;
use async_trait::async_trait;
use num_integer::div_ceil;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::any::type_name;
use std::error::Error;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io;
use std::io::{BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::debug;

pub struct MTStorage<F, MulTriples> {
    file: F,
    write_batch_size: usize,
    stored_mts: MulTriples,
    /// can only be set when feature bench-api is enabled
    insecure_loop_file: bool,
}

const DEFAULT_BATCH_SIZE: usize = 1_000;

impl<MulTriples> MTStorage<BufWriter<File>, MulTriples>
where
    MulTriples: Default,
{
    pub fn create(path: &Path) -> Result<Self, StorageError> {
        let file =
            File::create(path).map_err(|err| StorageError::CreateFile(path.to_path_buf(), err))?;
        Ok(Self {
            file: BufWriter::new(file),
            write_batch_size: DEFAULT_BATCH_SIZE,
            stored_mts: Default::default(),
            insecure_loop_file: false,
        })
    }
}

impl<F, MulTriples> MTStorage<F, MulTriples>
where
    MulTriples: Default,
{
    pub fn new(file: F) -> Self {
        Self {
            file,
            write_batch_size: DEFAULT_BATCH_SIZE,
            stored_mts: Default::default(),
            insecure_loop_file: false,
        }
    }

    /// WARNING: This is an insecure option. By setting loop_file = true, the mt storage
    /// file is simply read from the beginning when the end is reached. This is only intended for
    /// benchmarking, and thus only available with the "bench-api" feature enabled.
    #[cfg(feature = "bench-api")]
    pub fn insecure_loop_file(mut self, loop_file: bool) -> Self {
        self.insecure_loop_file = loop_file;
        self
    }
}
impl<F, MulTriples> MTStorage<F, MulTriples>
where
    F: Write + Debug,
{
    #[tracing::instrument(skip(self, mtp))]
    pub async fn store_mts<MTP>(&mut self, count: usize, mut mtp: MTP) -> Result<(), StorageError>
    where
        MTP: MTProvider<Output = MulTriples>,
        MTP::Error: Error + Send + Sync + 'static,
        MTP::Output: Serialize,
    {
        let batches = div_ceil(count, self.write_batch_size);
        for batch in 0..batches {
            debug!("Computing MT batch {batch}/{batches}");
            let mts = mtp
                .request_mts(self.write_batch_size)
                .await
                .map_err(|err| StorageError::MTProvider(Box::new(err)))?;
            bincode::serialize_into(&mut self.file, &mts).map_err(StorageError::MTSerialization)?;
        }
        Ok(())
    }
}

impl<F, MuLTriples> MTStorage<F, MuLTriples>
where
    F: Read + Debug,
    MuLTriples: DeserializeOwned,
{
    pub fn read_batch(&mut self) -> Result<MuLTriples, StorageError> {
        bincode::deserialize_from(&mut self.file).map_err(StorageError::MTDeserialization)
    }
}

#[async_trait]
impl<F: Read + Seek + Debug + Send, MulTriples> MTProvider for MTStorage<F, MulTriples>
where
    MulTriples: DeserializeOwned + SetupStorage,
{
    type Output = MulTriples;
    type Error = StorageError;

    async fn precompute_mts(&mut self, amount: usize) -> Result<(), Self::Error> {
        let mut added = 0;

        // TODO this calls append in a loop which will lead to frequent reallocations
        //  this could be mitigated by adding a reserve function to the SetupStorage trait
        self.stored_mts.reserve(amount);
        while added < amount {
            #[cfg(not(feature = "bench-api"))]
            let batch = self.read_batch()?;
            #[cfg(feature = "bench-api")]
            let batch = match self.read_batch() {
                Ok(batch) => batch,
                Err(StorageError::MTDeserialization(io))
                    if self.insecure_loop_file && matches!(*io, bincode::ErrorKind::Io(_)) =>
                {
                    self.rewind_file()?;
                    self.read_batch()?
                }
                Err(err) => {
                    return Err(err);
                }
            };
            added += batch.len();
            self.stored_mts.append(batch);
        }
        Ok(())
    }

    async fn request_mts(&mut self, amount: usize) -> Result<Self::Output, Self::Error> {
        if self.stored_mts.len() < amount {
            let missing = amount - self.stored_mts.len();
            self.precompute_mts(missing).await?;
        }
        Ok(self.stored_mts.remove_first(amount))
    }
}

impl<F, M> MTStorage<F, M> {
    /// Only relevant for writing
    pub fn batch_size(&self) -> usize {
        self.write_batch_size
    }
    /// Only used for writing MTs
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.write_batch_size = batch_size;
    }
}

impl<F: Seek, M> MTStorage<F, M> {
    pub fn rewind_file(&mut self) -> Result<(), StorageError> {
        self.file.rewind().map_err(StorageError::FileRewind)?;
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("creating MTs file at `{0}` failed")]
    CreateFile(PathBuf, #[source] io::Error),
    #[error("unable to generate MTs")]
    MTProvider(#[source] Box<dyn Error + Send + Sync + 'static>),
    #[error("unable to serialize MTs")]
    MTSerialization(#[source] bincode::Error),
    #[error("unable to deserialize MTs")]
    MTDeserialization(#[source] bincode::Error),
    #[error("unable to rewind MT file")]
    FileRewind(#[source] io::Error),
}

impl<F: Debug, M> Debug for MTStorage<F, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MTStorage")
            .field("file", &self.file)
            .field("batch_size", &self.write_batch_size)
            .field("storage_t", &type_name::<M>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::mul_triple::boolean::InsecureMTProvider;
    use crate::mul_triple::storage::MTStorage;
    use crate::mul_triple::MTProvider;
    use crate::private_test_utils::init_tracing;
    use std::io::Cursor;

    #[tokio::test]
    async fn mt_storage_precomp() {
        let _guard = init_tracing();
        let file = Cursor::new(vec![0_u8; 1500]);
        let mut mt_store = MTStorage::new(file);
        mt_store.set_batch_size(4);
        mt_store
            .store_mts(25, InsecureMTProvider::default())
            .await
            .unwrap();
        mt_store.rewind_file().unwrap();

        mt_store.precompute_mts(14).await.unwrap();
        let mts = mt_store.request_mts(22).await.unwrap();
        assert_eq!(22, mts.len());
    }
}
