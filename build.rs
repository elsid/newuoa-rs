fn main() {
    use std::process::Command;
    Command::new("cmake").args(&["."]).current_dir("newuoa-cpp").status().unwrap();
    Command::new("make").current_dir("newuoa-cpp").status().unwrap();
    println!("cargo:rustc-link-search=native=newuoa-cpp/lib");
    println!("cargo:rustc-link-lib=static=newuoa");
}
