import SwiftUI
import UIKit

/// Two halves: the source text is typed in the upper one, the translation
/// appears in the lower one while it is being generated.
struct ContentView: View {
    @State private var t = Translator()
    @State private var picking: Side?
    @FocusState private var editing: Bool

    enum Side: String, Identifiable {
        case source, target
        var id: String { rawValue }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                sourcePane
                    .frame(maxHeight: .infinity)
                divider
                targetPane
                    .frame(maxHeight: .infinity)
                footer
            }
            .toolbar(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { editing = false }
                }
            }
        }
        .task { await t.load() }
        .sheet(item: $picking) { side in
            LanguagePicker(
                languages: t.languages,
                selection: side == .source ? $t.srcLang : $t.tgtLang
            )
        }
    }

    // MARK: - upper half

    private var sourcePane: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                languageButton(t.srcLang, side: .source)
                Spacer()
                if !t.sourceText.isEmpty {
                    Button { t.clear() } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("Clear")
                }
            }
            ZStack(alignment: .topLeading) {
                TextEditor(text: $t.sourceText)
                    .focused($editing)
                    .font(.title3)
                    .scrollContentBackground(.hidden)
                if t.sourceText.isEmpty {
                    Text("Enter text")
                        .font(.title3)
                        .foregroundStyle(.tertiary)
                        .padding(.top, 8)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
    }

    // MARK: - the line between them

    private var divider: some View {
        ZStack {
            Divider()
            Button { t.swapLanguages() } label: {
                Image(systemName: "arrow.up.arrow.down")
                    .font(.body.weight(.semibold))
                    .frame(width: 40, height: 40)
                    .background(Circle().fill(Color(.secondarySystemBackground)))
                    .overlay(Circle().stroke(Color(.separator)))
            }
            .accessibilityLabel("Swap languages")
            .disabled(t.phase != .ready)
        }
        .frame(height: 40)
        .zIndex(1)
    }

    // MARK: - lower half

    private var targetPane: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                languageButton(t.tgtLang, side: .target)
                Spacer()
                if t.isTranslating {
                    ProgressView().controlSize(.small)
                }
                if !t.translation.isEmpty {
                    Button { UIPasteboard.general.string = t.translation } label: {
                        Image(systemName: "doc.on.doc")
                    }
                    .accessibilityLabel("Copy translation")
                }
            }
            ScrollView {
                Group {
                    switch t.phase {
                    case .ready:
                        Text(t.translation)
                            .font(.title3)
                            .textSelection(.enabled)
                    case .loading:
                        Label("Loading model…", systemImage: "hourglass")
                            .foregroundStyle(.secondary)
                    case .noModel:
                        Text("""
                            No model found.

                            Add a GGUF (e.g. nllb-600M-f16.gguf) either to the \
                            NllbTranslate folder in Xcode, or to this app's \
                            Documents via Finder → iPhone → Files, then relaunch.
                            """)
                            .foregroundStyle(.secondary)
                    case .failed(let message):
                        Text(message).foregroundStyle(.red)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
                .padding(.top, 8)
                .padding(.leading, 5)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .contentShape(Rectangle())
        .onTapGesture { editing = false }
    }

    private var footer: some View {
        HStack {
            Text(t.backend)
            Spacer()
            Text(t.stats)
        }
        .font(.caption2.monospacedDigit())
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(Color(.secondarySystemBackground))
    }

    private func languageButton(_ code: String, side: Side) -> some View {
        Button { picking = side } label: {
            HStack(spacing: 4) {
                Text(Languages.name(code))
                Image(systemName: "chevron.down").font(.caption)
            }
            .font(.headline)
        }
        .disabled(t.phase != .ready)
    }
}

/// Searchable list of every language in the loaded GGUF.
struct LanguagePicker: View {
    let languages: [String]
    @Binding var selection: String
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""

    private var filtered: [String] {
        let all = languages.sorted { Languages.name($0) < Languages.name($1) }
        guard !query.isEmpty else { return all }
        return all.filter {
            Languages.name($0).localizedCaseInsensitiveContains(query) ||
            $0.localizedCaseInsensitiveContains(query)
        }
    }

    var body: some View {
        NavigationStack {
            List {
                if query.isEmpty {
                    Section("Common") {
                        ForEach(Languages.common.filter { languages.contains($0) }, id: \.self, content: row)
                    }
                }
                Section(query.isEmpty ? "All \(languages.count)" : "Results") {
                    ForEach(filtered, id: \.self, content: row)
                }
            }
            .searchable(text: $query, placement: .navigationBarDrawer(displayMode: .always))
            .navigationTitle("Language")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }

    private func row(_ code: String) -> some View {
        Button {
            selection = code
            dismiss()
        } label: {
            HStack {
                VStack(alignment: .leading) {
                    Text(Languages.name(code)).foregroundStyle(.primary)
                    Text(code).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                if code == selection {
                    Image(systemName: "checkmark").foregroundStyle(.tint)
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
