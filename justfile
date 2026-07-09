fmt:
    cargo fmt --all

lint:
    cargo clippy --all-targets --all-features -- -D warnings

check:
    cargo check --all-features

test:
    cargo nextest run --all-features

doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

ci:
    just fmt
    just lint
    just check
    just test
    cargo test --doc
    just doc
