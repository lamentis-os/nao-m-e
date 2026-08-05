use nao_m_e_semantic::{
    CueText, E5_SMALL_PROFILE, EMBEDDING_DIMENSIONS, Embedding, MAX_EMBEDDING_BATCH_SIZE,
    SemanticEncoder, SemanticError,
};

const FIXTURE_OUTPUT: &str = "NAO_M_E_SEMANTIC_FIXTURE_PATH";
const PROBLEM_CUE_GOLDEN: [i16; EMBEDDING_DIMENSIONS] = [
    1758, 672, -766, -1645, 1994, -981, 959, 401, 2285, 283, 1319, 832, 1849, -1529, -1842, 493,
    2803, -787, -1308, -1720, 1126, 721, -1307, 1171, 1826, 1948, 190, 2141, 1846, -3544, -1428,
    -101, 1988, -1799, 2051, 1986, -1189, -1695, 1946, -2898, -509, 489, 760, 2841, 843, -64,
    -1250, 1879, -1053, 102, -1836, 1657, -364, 2147, 1590, -1481, -976, -1983, -2665, 710, 1580,
    -58, 854, 520, 3887, 2578, 856, 1906, -1878, -1550, -2193, 1731, -128, -1006, 323, 672, 560,
    -1139, 1637, -1031, -1110, -1096, -764, 1796, -1008, 1377, 1431, -1753, 3390, -1263, 3058,
    2373, -1427, -2111, -1557, -2345, -2648, 2701, 1405, -424, 1841, -1838, 1853, -2841, -2687,
    2301, 721, -1628, 2023, -2398, -2657, 972, 1270, 409, -2079, -1592, -925, -1820, 1135, -2271,
    1390, 536, -1029, -2100, -1835, -1780, 1002, 2261, -267, 406, 1777, 964, 1067, 1844, 1069,
    2594, -1077, 146, -435, -1245, -2014, 1556, -1286, 1249, 1464, 803, 3003, -170, 2729, -2322,
    3011, -2398, 2440, 829, 1660, -2273, -1766, -1511, 1797, 957, -2171, -1923, -2075, -540, -1980,
    -1870, -633, 2825, -2502, -327, -1494, 1488, -630, 2684, 76, 1296, -1072, 1478, 2914, 642,
    -372, -1599, -1945, -1414, -2010, -2079, -2145, 257, 746, 66, -957, 877, -997, -1566, -615,
    310, -1492, 2258, 2199, 2022, -331, -664, 958, 1588, 1917, -317, -1347, 1004, -1290, 358, 1728,
    -1078, -3075, 1877, -1895, -112, 1379, 2337, -762, 1920, 804, -823, 976, -3017, -2241, 1278,
    2230, -2421, -3269, 798, -1575, -1175, -1515, -2363, -1924, -2225, 911, 876, 53, -534, -991,
    -1122, -191, -1710, 1791, -480, -365, 1078, 182, 2706, 2335, -3237, -1905, -1813, -447, 1630,
    2603, 2970, -964, 809, 877, -1227, 2305, 1243, 1988, 1219, -2104, -1317, -1781, -2617, -1192,
    588, 1744, -3536, 1010, -2218, 241, 1630, -1465, -2298, 1685, -119, 1641, 2089, 2371, -589,
    -315, 121, -1318, -1504, -628, -1584, -499, -3313, 2043, 1422, 122, 1550, -1336, 1287, -222,
    -1420, 1311, -301, -3922, 1538, 1575, 1505, 1335, 1093, 2134, 2231, -1570, -1680, -143, 3070,
    899, 491, -1163, -2689, -1935, -1667, 191, -1278, 1257, 2352, -1408, -1316, 2235, 1144, 641,
    -492, -1524, 823, -1414, 303, -1454, 816, -2468, -891, 450, 1528, -2154, -110, -775, -2662,
    1451, -657, -2156, 1346, 2006, -2508, -240, 1681, -1023, 2830, -3175, -846, 1588, 679, -3360,
    -1100, -133, 2525, 1942, 2088, -424, -873, 546, -2366, 2178, 593, -1226, -286, -1293, -2064,
    -2072, 1702, -893, -1593, 1630, 1804, 732, 1433,
];

#[test]
fn public_fixed_profile_and_embedding_contract_are_coherent() {
    assert_eq!(E5_SMALL_PROFILE.dimensions(), EMBEDDING_DIMENSIONS);
    assert!(!E5_SMALL_PROFILE.manifest().is_empty());
    assert_ne!(E5_SMALL_PROFILE.fingerprint(), [0; 32]);

    assert!(Embedding::new(vec![1; EMBEDDING_DIMENSIONS - 1]).is_none());
    assert!(Embedding::new(vec![i16::MIN; EMBEDDING_DIMENSIONS]).is_none());
    let embedding = Embedding::new(vec![1; EMBEDDING_DIMENSIONS]).unwrap();
    assert_eq!(embedding.profile(), E5_SMALL_PROFILE);
}

#[test]
fn public_encoder_is_lazy_for_non_runtime_paths() {
    let mut encoder = SemanticEncoder::new();
    assert!(!encoder.is_loaded());
    assert!(encoder.encode(&[]).unwrap().is_empty());

    let cues = vec![CueText::new("key", "value"); MAX_EMBEDDING_BATCH_SIZE + 1];
    assert!(matches!(
        encoder.encode(&cues),
        Err(SemanticError::BatchTooLarge { .. })
    ));
    assert!(!encoder.is_loaded());
}

#[test]
#[ignore = "downloads and executes the pinned 470 MB E5 Small model"]
fn pinned_runtime_smoke_is_explicit_and_repeatable() {
    let long_value = std::iter::repeat_n("grenze", 700)
        .collect::<Vec<_>>()
        .join(" ");
    let values = [
        ("problem", "login returns http 404".to_owned()),
        ("activity", "hund am strand spazieren".to_owned()),
        ("unicode", "straße ångström 東京 🐕 café שלום".to_owned()),
        ("truncation", long_value),
    ];
    let cues: Vec<_> = values
        .iter()
        .map(|(key, value)| CueText::new(key, value))
        .collect();
    let mut encoder = SemanticEncoder::new();
    let embeddings = encoder.encode(&cues).unwrap();
    assert!(encoder.is_loaded());
    assert_eq!(embeddings.len(), cues.len());
    assert!(
        embeddings
            .iter()
            .all(|embedding| embedding.values().len() == EMBEDDING_DIMENSIONS)
    );
    assert!(
        embeddings[0]
            .values()
            .iter()
            .zip(PROBLEM_CUE_GOLDEN)
            .all(|(actual, expected)| i32::from(*actual).abs_diff(i32::from(expected)) <= 1),
        "fixed problem cue drifted outside the cross-platform profile tolerance"
    );

    for (cue, expected) in cues.iter().zip(&embeddings) {
        let singleton = encoder.encode(std::slice::from_ref(cue)).unwrap();
        assert_eq!(&singleton[0], expected);
    }

    let mut reversed = cues.clone();
    reversed.reverse();
    let mut reversed_embeddings = encoder.encode(&reversed).unwrap();
    reversed_embeddings.reverse();
    assert_eq!(reversed_embeddings, embeddings);

    if let Some(path) = std::env::var_os(FIXTURE_OUTPUT) {
        std::fs::write(path, fixture_bytes(&embeddings)).unwrap();
    }
}

fn fixture_bytes(embeddings: &[Embedding]) -> Vec<u8> {
    let component_bytes = embeddings
        .len()
        .checked_mul(EMBEDDING_DIMENSIONS)
        .and_then(|value| value.checked_mul(size_of::<i16>()))
        .unwrap();
    let mut bytes = Vec::with_capacity(24 + 32 + component_bytes);
    bytes.extend_from_slice(b"nao-m-e-e5-fixture-v1\0");
    bytes.extend_from_slice(&E5_SMALL_PROFILE.fingerprint());
    bytes.extend_from_slice(&u32::try_from(embeddings.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&u16::try_from(EMBEDDING_DIMENSIONS).unwrap().to_le_bytes());
    for embedding in embeddings {
        for value in embedding.values() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}
