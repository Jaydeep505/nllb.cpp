import Foundation
import Observation

/// App state: the source text, the two languages, and the translation that
/// follows the text as it is typed.
///
/// Every edit cancels the running translation (it stops at the next token)
/// and schedules a new one after a short pause, so the model only ever works
/// on what is currently on screen.
@Observable
@MainActor
final class Translator {
    enum Phase: Equatable {
        case loading
        case noModel
        case ready
        case failed(String)
    }

    private(set) var phase: Phase = .loading
    private(set) var translation = ""
    private(set) var isTranslating = false
    private(set) var stats = ""
    private(set) var backend = ""
    private(set) var languages: [String] = []

    var sourceText = "" {
        didSet { if sourceText != oldValue { schedule() } }
    }
    var srcLang: String = UserDefaults.standard.string(forKey: "srcLang") ?? "eng_Latn" {
        didSet { UserDefaults.standard.set(srcLang, forKey: "srcLang"); if !swapping { schedule(immediate: true) } }
    }
    var tgtLang: String = UserDefaults.standard.string(forKey: "tgtLang") ?? "deu_Latn" {
        didSet { UserDefaults.standard.set(tgtLang, forKey: "tgtLang"); if !swapping { schedule(immediate: true) } }
    }

    /// Pause after the last keystroke before translating.
    private static let debounce: Duration = .milliseconds(350)

    @ObservationIgnored private var ctx: NllbContext?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var requestID = 0
    @ObservationIgnored private var swapping = false

    // MARK: - model

    func load() async {
        guard ctx == nil else { return }
        guard let url = ModelLocator.find() else {
            phase = .noModel
            return
        }
        phase = .loading
        do {
            // Metal on a real device. The simulator's Metal is a translation
            // layer over the Mac's GPU: nothing measured there transfers, and
            // ggml's kernels are not verified on it - so it runs on the CPU.
            #if targetEnvironment(simulator)
            let backendChoice = "cpu"
            #else
            let backendChoice = "auto"
            #endif
            let c = try await NllbContext.load(path: url.path, backend: backendChoice)
            ctx = c
            languages = c.languages
            backend = "\(c.backendName)\(c.zeroCopy ? " · zero-copy" : "") · \(url.lastPathComponent)"
            if !languages.contains(srcLang) { srcLang = "eng_Latn" }
            if !languages.contains(tgtLang) { tgtLang = "deu_Latn" }
            phase = .ready
            schedule(immediate: true)
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    // MARK: - actions

    func swapLanguages() {
        swapping = true
        (srcLang, tgtLang) = (tgtLang, srcLang)
        swapping = false
        if !translation.isEmpty && !isTranslating {
            sourceText = translation      // schedules via didSet
        } else {
            schedule(immediate: true)
        }
    }

    func clear() {
        sourceText = ""
    }

    // MARK: - translation

    private func schedule(immediate: Bool = false) {
        task?.cancel()
        requestID += 1
        let id = requestID

        guard phase == .ready, let ctx else { return }

        let text = sourceText
        let src = srcLang, tgt = tgtLang
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            translation = ""
            isTranslating = false
            stats = ""
            return
        }

        // Called on the inference thread after every token. The main queue
        // is FIFO, so partials arrive in order; stale ones are dropped.
        let onPartial: @Sendable (String) -> Void = { [weak self] partial in
            DispatchQueue.main.async { [weak self] in
                MainActor.assumeIsolated {
                    guard let self, self.requestID == id else { return }
                    self.translation = partial
                }
            }
        }

        task = Task { [weak self] in
            if !immediate {
                try? await Task.sleep(for: Translator.debounce)
            }
            guard !Task.isCancelled else { return }
            self?.isTranslating = true

            let clock = ContinuousClock()
            let start = clock.now
            do {
                let result = try await ctx.translate(text, from: src, to: tgt, onPartial: onPartial)
                guard let self, self.requestID == id else { return }
                let elapsed = start.duration(to: clock.now)
                let secs = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18
                self.translation = result.text
                self.stats = String(format: "%ld tokens · %.2f s · %.1f tok/s",
                                    result.tokens, secs, secs > 0 ? Double(result.tokens) / secs : 0)
                self.isTranslating = false
            } catch is CancellationError {
                // superseded by a newer request; it owns the UI now
            } catch {
                guard let self, self.requestID == id else { return }
                self.stats = error.localizedDescription
                self.isTranslating = false
            }
        }
    }
}

/// Where the GGUF comes from, in order:
///  1. the app bundle - drop the file into the NllbTranslate/ folder and it is
///     copied in as a resource (simplest for development);
///  2. the app's Documents folder - copy it there with Finder (iPhone ->
///     Files -> NllbTranslate) or the Files app, so a 1.2 GB file is not
///     re-installed with every build.
/// Among several files, f16 is preferred (the only mix with headroom > 1),
/// then q8emb.
enum ModelLocator {
    static func find() -> URL? {
        var candidates: [URL] = []
        if let urls = Bundle.main.urls(forResourcesWithExtension: "gguf", subdirectory: nil) {
            candidates += urls
        }
        if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
           let items = try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil) {
            candidates += items.filter { $0.pathExtension.lowercased() == "gguf" }
        }
        func rank(_ u: URL) -> Int {
            let n = u.lastPathComponent.lowercased()
            if n.contains("f16")   { return 0 }
            if n.contains("q8emb") { return 1 }
            return 2
        }
        return candidates.sorted { (rank($0), $0.lastPathComponent) < (rank($1), $1.lastPathComponent) }.first
    }
}
