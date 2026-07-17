//! Kimi-K3 tokenizer loading against a real model snapshot.
//!
//! Env-gated: skipped unless `K3_SNAPSHOT_DIR` points at a K3 model dir
//! (config.json + tokenizer_config.json + tiktoken.model), e.g.
//! `K3_SNAPSHOT_DIR=~/repos/K3-prod/configs cargo test -p dynamo-llm --test kimi_k3_tokenizer`

use dynamo_llm::model_card::ModelDeploymentCard;

const OPEN_ID: u32 = 163587;
const SEP_ID: u32 = 163589;
const END_OF_MSG_ID: u32 = 163586;

fn snapshot_dir() -> Option<std::path::PathBuf> {
    let dir = std::env::var_os("K3_SNAPSHOT_DIR")?;
    let dir = std::path::PathBuf::from(dir);
    let dir = if let Ok(stripped) = dir.strip_prefix("~") {
        std::path::PathBuf::from(std::env::var("HOME").expect("HOME set")).join(stripped)
    } else {
        dir
    };
    Some(dir)
}

#[test]
fn k3_snapshot_loads_and_markers_are_atomic() {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipped: K3_SNAPSHOT_DIR unset");
        return;
    };

    let mdc = ModelDeploymentCard::load_from_disk(&dir, None).expect("load K3 MDC");
    let tokenizer = mdc.tokenizer().expect("build K3 tokenizer");

    // Structural markers are registered specials: each encodes to a single
    // atomic id inside the marker sequence.
    let text = "<|open|>think<|sep|>";
    let encoding = tokenizer.encode(text).expect("encode");
    let ids = encoding.token_ids().to_vec();
    assert!(
        ids.contains(&OPEN_ID),
        "expected atomic <|open|> id {OPEN_ID} in {ids:?}"
    );
    assert!(
        ids.contains(&SEP_ID),
        "expected atomic <|sep|> id {SEP_ID} in {ids:?}"
    );

    // Round-trip: decoding (keeping specials) reproduces the exact text.
    let decoded = tokenizer.decode(&ids, false).expect("decode");
    assert_eq!(decoded.as_str(), text);

    // The generation eos (<|end_of_msg|> = 163586, from generation_config.json)
    // must surface through the model-info eos chain.
    let model_info = mdc
        .model_info
        .as_ref()
        .expect("model_info present")
        .get_model_info()
        .expect("parse model info");
    let eos = model_info.eos_token_ids();
    assert!(
        eos.contains(&END_OF_MSG_ID),
        "expected eos {END_OF_MSG_ID} in {eos:?}"
    );
}
