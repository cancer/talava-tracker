fn main() {
    // Rerun when git HEAD changes (commit, checkout, etc.)
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");

    let output = std::process::Command::new("git")
        .args(["describe", "--always", "--dirty", "--tags"])
        .output();

    let version = match output {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout).trim().to_string()
        }
        _ => "unknown".to_string(),
    };

    println!("cargo:rustc-env=GIT_VERSION={}", version);
}
