import SwiftUI

@main
struct CellmDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ZStack {
                Color(.systemGroupedBackground).ignoresSafeArea()
                TabView {
                    ModelsView()
                        .tabItem {
                            Label("Models Hub", systemImage: "arrow.down.doc")
                        }
                    LLMView()
                        .tabItem {
                            Label("LLM Sandbox", systemImage: "text.bubble")
                        }
                    VLMView()
                        .tabItem {
                            Label("Vision", systemImage: "photo.on.rectangle")
                        }
                    ChatView()
                        .tabItem {
                            Label("Chat Bot", systemImage: "bubble.left.and.bubble.right")
                        }
                }
            }
        }
    }
}
