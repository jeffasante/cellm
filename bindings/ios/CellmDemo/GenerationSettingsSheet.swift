import SwiftUI

struct GenerationSettingsSheet: View {
    @Binding var temperature: Double
    @Binding var maxNewTokens: Int
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("Temperature", systemImage: "thermometer.medium")
                            Spacer()
                            Text(String(format: "%.2f", temperature))
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $temperature, in: 0...2, step: 0.05)
                            .tint(.orange)
                        Text(temperatureHint)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text("Sampling")
                }

                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("Max Tokens", systemImage: "textformat.123")
                            Spacer()
                            Text("\(maxNewTokens)")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        Slider(
                            value: Binding(
                                get: { Double(maxNewTokens) },
                                set: { maxNewTokens = Int($0.rounded()) }
                            ),
                            in: 50...512,
                            step: 10
                        )
                        .tint(.blue)
                        Text("Maximum number of tokens the model will generate per response.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text("Generation Length")
                }

                Section {
                    Button("Reset to Defaults") {
                        withAnimation {
                            temperature = 0.2
                            maxNewTokens = 200
                        }
                    }
                    .foregroundStyle(.red)
                }
            }
            .navigationTitle("Generation Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium])
        .presentationDragIndicator(.visible)
    }

    private var temperatureHint: String {
        switch temperature {
        case 0..<0.05:  return "Greedy (deterministic, most accurate)"
        case 0.05..<0.4: return "Focused (low creativity, factual)"
        case 0.4..<0.8: return "Balanced (default)"
        case 0.8..<1.2: return "Creative (more variety)"
        default:        return "Very creative (may produce unexpected output)"
        }
    }
}

#Preview {
    GenerationSettingsSheet(temperature: .constant(0.2), maxNewTokens: .constant(200))
}
