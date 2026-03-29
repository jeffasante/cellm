// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "CellmSDK",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "CellmSDK", targets: ["CellmSDK"]),
    ],
    targets: [
        .target(
            name: "CellmSDK",
            dependencies: ["CellmFFI"]
        ),
        .binaryTarget(
            name: "CellmFFI",
            path: "CellmFFI.xcframework"
        )
    ]
)
