import SwiftUI

@main
struct CellmDemoApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                LLMView()
                    .tabItem { Text("LLM") }
                VLMView()
                    .tabItem { Text("VLM") }
            }
        }
    }
}

