use rand::rngs::StdRng;
use rand::SeedableRng;

/// Create a seeded RNG, or a random one if no seed provided.
pub fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    }
}
