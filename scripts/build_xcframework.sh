#!/bin/zsh -euo pipefail

# Builds an Apple XCFramework for the `cellm-sdk` staticlib.
#
# Outputs:
#   bindings/swift/CellmFFI.xcframework
#
# Requirements:
#   - Xcode command line tools
#   - Rust targets installed:
#       rustup target add aarch64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/bindings/swift"
HDR_DIR="$ROOT_DIR/crates/cellm-sdk/include"

cd "$ROOT_DIR"

echo "==> Building macOS (arm64)"
cargo rustc --release -p cellm-sdk --target aarch64-apple-darwin --lib --crate-type staticlib

echo "==> Building iOS device (arm64)"
cargo rustc --release -p cellm-sdk --target aarch64-apple-ios --lib --crate-type staticlib

echo "==> Building iOS simulator (arm64)"
cargo rustc --release -p cellm-sdk --target aarch64-apple-ios-sim --lib --crate-type staticlib

MAC_LIB="$ROOT_DIR/target/aarch64-apple-darwin/release/libcellm_sdk.a"
IOS_LIB="$ROOT_DIR/target/aarch64-apple-ios/release/libcellm_sdk.a"
SIM_LIB="$ROOT_DIR/target/aarch64-apple-ios-sim/release/libcellm_sdk.a"

if [[ ! -f "$MAC_LIB" ]]; then
  echo "missing macOS staticlib: $MAC_LIB" >&2
  exit 1
fi
if [[ ! -f "$IOS_LIB" ]]; then
  echo "missing iOS device staticlib: $IOS_LIB" >&2
  exit 1
fi
if [[ ! -f "$SIM_LIB" ]]; then
  echo "missing iOS simulator staticlib: $SIM_LIB" >&2
  exit 1
fi

XCFRAMEWORK="$OUT_DIR/CellmFFI.xcframework"
rm -rf "$XCFRAMEWORK"

echo "==> Creating XCFramework"
xcodebuild -create-xcframework \
  -library "$MAC_LIB" -headers "$HDR_DIR" \
  -library "$IOS_LIB" -headers "$HDR_DIR" \
  -library "$SIM_LIB" -headers "$HDR_DIR" \
  -output "$XCFRAMEWORK"

echo "==> Wrote $XCFRAMEWORK"
