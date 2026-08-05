use nao_m_e_semantic::{
    E5_SMALL_PROFILE, EMBEDDING_DIMENSIONS, Embedding, EpisodeText, QueryText, SemanticEncoder,
    SemanticError,
};

const SCALE: u128 = 1_000_000;

struct JudgedEpisode {
    attributes: &'static [(&'static str, &'static str)],
}

struct JudgedQuery {
    text: &'static str,
    expected: usize,
}

const EPISODES: &[JudgedEpisode] = &[
    JudgedEpisode {
        attributes: &[
            ("component", "authentication"),
            ("event", "login request returned http 404"),
            ("project", "lamentis"),
            ("status", "failed"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("component", "authentication"),
            ("event", "login request returned http 401 unauthorized"),
            ("project", "lamentis"),
            ("status", "failed"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("component", "authentication"),
            ("event", "login request returned http 404"),
            ("project", "orion"),
            ("status", "failed"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("component", "checkout"),
            ("event", "checkout request returned http 404"),
            ("project", "lamentis"),
            ("status", "failed"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("activity", "walked with the dog on the beach"),
            ("context", "sunny afternoon"),
            ("place", "beach"),
            ("type", "personal memory"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("activity", "walked with the dog on the beach"),
            ("context", "rainy afternoon"),
            ("place", "beach"),
            ("type", "personal memory"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("activity", "walked with the dog in the park"),
            ("context", "rainy afternoon"),
            ("place", "park"),
            ("type", "personal memory"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("problem", "linker cannot find sqlite library"),
            ("project", "rust workspace"),
            ("status", "failed"),
            ("tool", "cargo build"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("problem", "borrow checker rejects mutable reference"),
            ("project", "rust workspace"),
            ("status", "failed"),
            ("tool", "cargo check"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("action", "cooked pasta with tomato sauce"),
            ("context", "dinner"),
            ("subject", "pasta"),
            ("type", "personal memory"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("action", "planted tomato seedlings in the garden"),
            ("context", "weekend"),
            ("subject", "gardening"),
            ("type", "personal memory"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("genre", "fictional movie"),
            (
                "plot",
                "a dog walks on a beach while hackers investigate a lamentis login http 404",
            ),
            ("title", "the signal shore"),
            ("type", "movie synopsis"),
        ],
    },
    JudgedEpisode {
        attributes: &[
            ("activity", "renewed a passport"),
            ("context", "weekday appointment"),
            ("place", "municipal office"),
            ("subject", "travel document"),
        ],
    },
];

const QUERIES: &[JudgedQuery] = &[
    JudgedQuery {
        text: "login request in lamentis returns http 404",
        expected: 0,
    },
    JudgedQuery {
        text: "login request in orion returns http 404",
        expected: 2,
    },
    JudgedQuery {
        text: "checkout request in lamentis returns http 404",
        expected: 3,
    },
    JudgedQuery {
        text: "personal memory walking the dog on the beach on a rainy afternoon",
        expected: 5,
    },
    JudgedQuery {
        text: "personal memory walking the dog on the beach on a sunny afternoon",
        expected: 4,
    },
    JudgedQuery {
        text: "rust linker cannot find sqlite library during cargo build",
        expected: 7,
    },
    JudgedQuery {
        text: "cooked pasta with tomato sauce for dinner",
        expected: 9,
    },
    JudgedQuery {
        text: "planted tomato seedlings in the garden",
        expected: 10,
    },
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
#[ignore = "requires and executes the provisioned pinned 470 MB E5 Small model"]
fn pinned_runtime_passes_the_predeclared_episode_retrieval_gate() {
    let mut encoder = SemanticEncoder::new();
    let episodes = EPISODES
        .iter()
        .map(|episode| {
            encoder
                .encode_episode(EpisodeText::new(episode.attributes).unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();

    for query in QUERIES {
        let query_embedding = encoder.encode_query(QueryText::new(query.text)).unwrap();
        assert_eq!(
            encoder.encode_query(QueryText::new(query.text)).unwrap(),
            query_embedding,
            "query encoding is not repeatable for {:?}",
            query.text
        );
        let ranking = rank(&query_embedding, &episodes);
        eprintln!(
            "query={:?} expected={} ranking={:?}",
            query.text,
            query.expected,
            &ranking[..3]
        );
        assert_eq!(
            ranking[0].0, query.expected,
            "predeclared exact episode did not rank first for {:?}",
            query.text
        );
    }

    let first = EpisodeText::new(EPISODES[0].attributes).unwrap();
    assert_eq!(encoder.encode_episode(first).unwrap(), episodes[0]);

    let mut long_value = String::with_capacity(9 * 700);
    for index in 0..700 {
        if index != 0 {
            long_value.push(' ');
        }
        long_value.push_str("boundary");
    }
    let long_attributes = [("content", long_value.as_str())];
    assert!(matches!(
        encoder.encode_episode(EpisodeText::new(&long_attributes).unwrap()),
        Err(SemanticError::EpisodeTooLong { maximum: 512 })
    ));
    assert!(
        encoder.encode_query(QueryText::new(&long_value)).is_ok(),
        "query overflow must remain right-truncated"
    );
}

fn rank(query: &Embedding, episodes: &[Embedding]) -> Vec<(usize, u32)> {
    let mut ranking = episodes
        .iter()
        .enumerate()
        .map(|(sequence, episode)| (sequence, cosine_ppm(query.values(), episode.values())))
        .collect::<Vec<_>>();
    ranking.sort_unstable_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    ranking
}

fn cosine_ppm(query: &[i16], episode: &[i16]) -> u32 {
    let mut dot = 0_i64;
    let mut query_norm = 0_u64;
    let mut episode_norm = 0_u64;
    for (&query, &episode) in query.iter().zip(episode) {
        let query = i64::from(query);
        let episode = i64::from(episode);
        dot += query * episode;
        query_norm += u64::try_from(query * query).unwrap();
        episode_norm += u64::try_from(episode * episode).unwrap();
    }
    if dot <= 0 {
        return 0;
    }
    let denominator = (u128::from(query_norm) * u128::from(episode_norm)).isqrt();
    let score = u128::try_from(dot).unwrap() * SCALE / denominator;
    u32::try_from(score.min(SCALE)).unwrap()
}
