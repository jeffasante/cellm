// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "CellmSDK",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "CellmSDK", targets: ["CellmSDK"]),
        .executable(name: "CellmSmoke", targets: ["CellmSmoke"]),
    ],
    targets: [
        .target(
            name: "CellmSDK",
            dependencies: ["CellmFFI"]
        ),
        .executableTarget(
            name: "CellmSmoke",
            dependencies: ["CellmSDK"],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedLibrary("c++")
            ]
        ),
        .binaryTarget(
            name: "CellmFFI",
            path: "CellmFFI.xcframework"
        )
    ]
)
