use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Create a seeded RNG, or a random one if no seed provided.
pub fn make_rng(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_os_rng(),
    }
}
