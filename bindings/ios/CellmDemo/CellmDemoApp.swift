import SwiftUI

@main
struct CellmDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ZStack {
                Color(.systemGroupedBackground).ignoresSafeArea()
                TabView {
                    LLMView()
                        .tabItem {
                            Label("LLM", systemImage: "text.bubble")
                        }
                    VLMView()
                        .tabItem {
                            Label("VLM", systemImage: "photo.on.rectangle")
                        }
                }
            }
        }
    }
}
