use remoc::codec;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fmt,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestEnum {
    One(u16),
    Two { field1: String, field2: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestStruct {
    simple: String,
    btree: BTreeMap<Vec<u8>, String>,
    hash: HashMap<(u16, String), u8>,
    enu: Vec<TestEnum>,
}

impl Default for TestStruct {
    fn default() -> Self {
        let mut data = Self {
            simple: "test_string".to_string(),
            btree: BTreeMap::new(),
            hash: HashMap::new(),
            enu: vec![TestEnum::One(11), TestEnum::Two { field1: "value1".to_string(), field2: 2 }],
        };
        data.btree.insert(vec![1, 2, 3], "first value".to_string());
        data.btree.insert(vec![4, 5, 6, 7], "second value".to_string());
        data.hash.insert((1, "one".to_string()), 10);
        data.hash.insert((2, "two".to_string()), 20);
        data.hash.insert((3, "three".to_string()), 30);
        data
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestStructWithAttr {
    simple: String,
    #[serde(with = "remoc::codec::map::btreemap")]
    btree: BTreeMap<Vec<u8>, String>,
    #[serde(with = "remoc::codec::map::hashmap")]
    hash: HashMap<(u16, String), u8>,
    enu: Vec<TestEnum>,
}

impl Default for TestStructWithAttr {
    fn default() -> Self {
        let mut data = Self {
            simple: "test_string".to_string(),
            btree: BTreeMap::new(),
            hash: HashMap::new(),
            enu: vec![TestEnum::One(11), TestEnum::Two { field1: "value1".to_string(), field2: 2 }],
        };
        data.btree.insert(vec![1, 2, 3], "first value".to_string());
        data.btree.insert(vec![4, 5, 6, 7], "second value".to_string());
        data.hash.insert((1, "one".to_string()), 10);
        data.hash.insert((2, "two".to_string()), 20);
        data.hash.insert((3, "three".to_string()), 30);
        data
    }
}

#[allow(dead_code)]
fn roundtrip<T, Codec>()
where
    T: Default + Serialize + DeserializeOwned + fmt::Debug + Eq,
    Codec: codec::Codec,
{
    let data: T = Default::default();
    println!("data:\n{:?}", &data);

    let mut buffer = Vec::new();
    <Codec as codec::Codec>::serialize(&mut buffer, &data).unwrap();
    println!("serialized ({} bytes):\n{}", buffer.len(), String::from_utf8_lossy(&buffer));

    let deser: T = <Codec as codec::Codec>::deserialize(buffer.as_slice()).unwrap();
    assert_eq!(deser, data);
}

#[cfg(feature = "codec-bincode")]
#[test]
fn bincode() {
    roundtrip::<TestStruct, codec::Bincode>()
}

#[cfg(feature = "codec-cbor")]
#[test]
fn cbor() {
    #[allow(deprecated)]
    roundtrip::<TestStruct, codec::Cbor>()
}

#[cfg(feature = "codec-ciborium")]
#[test]
fn ciborium() {
    roundtrip::<TestStruct, codec::Ciborium>()
}

#[cfg(feature = "codec-json")]
#[test]
#[should_panic]
fn json_without_attr() {
    roundtrip::<TestStruct, codec::Json>()
}

#[cfg(feature = "codec-json")]
#[test]
fn json_with_attr() {
    roundtrip::<TestStructWithAttr, codec::Json>()
}

#[cfg(feature = "codec-message-pack")]
#[test]
fn message_pack() {
    roundtrip::<TestStruct, codec::MessagePack>()
}
