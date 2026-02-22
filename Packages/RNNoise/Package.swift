// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RNNoise",
    products: [
        .library(name: "CRNNoise", targets: ["CRNNoise"])
    ],
    targets: [
        .target(
            name: "CRNNoise",
            path: "Sources/CRNNoise",
            publicHeadersPath: "include",
            cSettings: [
                .define("RNNOISE_EXPORT", to: ""),
            ]
        )
    ]
)
