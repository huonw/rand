`#[derive]`-like functionality for the `rand::Rand` trait.

## Example

```rust
#![feature(plugin)]

#[plugin] #[no_link] extern crate derive_rand;
extern crate rand;

#[derive_Rand]
struct Foo {
    x: u8,
    y: isize
}

#[derive_Rand]
enum Bar {
    X(char),
    Y(f64)
}
```
