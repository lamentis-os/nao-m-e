use nao_m_e_sqlite::SqliteStore;

#[test]
fn public_create_noop_save_check_and_reopen_need_no_model() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("memory.sqlite3");
    let mut store = SqliteStore::create(&path).unwrap();
    let memory_id = store.memory_id();

    store.save().unwrap();
    drop(store);
    let before = std::fs::read(&path).unwrap();
    SqliteStore::check(&path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);

    let reopened = SqliteStore::open(&path).unwrap();
    assert_eq!(reopened.memory_id(), memory_id);
    assert_eq!(reopened.memory().episodes().len(), 0);
}
