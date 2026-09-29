import Foundation
import nllb

enum NllbError: LocalizedError {
    case load(String)
    case translate(String)

    var errorDescription: String? {
        switch self {
        case .load(let m):      return m
        case .translate(let m): return m
        }
    }
}

/// Swift wrapper around nllb-c.h.
///
/// One `nllb_ctx` must only be used from one thread at a time, so every call
/// into it goes through `queue`, a private serial queue. That also keeps the
/// multi-second blocking C call off Swift's cooperative thread pool.
final class NllbContext: @unchecked Sendable {
    private let ctx: OpaquePointer
    private let queue = DispatchQueue(label: "nllb.inference", qos: .userInitiated)

    /// ggml's device name, e.g. "MTL0" on the GPU or "CPU".
    let backendName: String
    /// True when the mmap'ed weights are wrapped zero-copy in a Metal buffer.
    let zeroCopy: Bool
    /// FLORES-200 codes carried by the GGUF, in id order.
    let languages: [String]

    private init(ctx: OpaquePointer) {
        self.ctx = ctx
        backendName = String(cString: nllb_ctx_backend_name(ctx))
        zeroCopy = nllb_ctx_weights_zero_copy(ctx) != 0
        languages = (0..<nllb_ctx_n_langs(ctx)).compactMap { i in
            nllb_ctx_lang(ctx, i).map { String(cString: $0) }
        }
    }

    deinit { nllb_ctx_free(ctx) }

    /// Loads a GGUF. "auto" = Metal when present, otherwise the CPU.
    static func load(path: String, backend: String = "auto") async throws -> NllbContext {
        try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                var params = nllb_ctx_default_params()
                params.use_mmap = 1
                // On Metal the CPU threads only run the ops Metal lacks.
                params.n_threads = Int32(max(1, min(4, ProcessInfo.processInfo.activeProcessorCount - 2)))

                var err = [CChar](repeating: 0, count: 1024)
                let errLen = err.count
                let ptr: OpaquePointer? = backend.withCString { b in
                    params.backend = b
                    return path.withCString { nllb_ctx_init($0, params, &err, errLen) }
                }
                if let ptr {
                    cont.resume(returning: NllbContext(ctx: ptr))
                } else {
                    cont.resume(throwing: NllbError.load(NllbContext.string(err)))
                }
            }
        }
    }

    struct Result {
        let text: String
        let tokens: Int
        let aborted: Bool
    }

    /// Translates `text` line by line. `onPartial` receives the whole
    /// translation-so-far after every generated token, on the inference
    /// thread. Cancelling the calling Task stops generation at the next token.
    func translate(_ text: String, from src: String, to tgt: String,
                   maxNewTokens: Int = 256,
                   onPartial: @escaping @Sendable (String) -> Void) async throws -> Result {
        let box = PartialBox(onPartial)
        let ctx = self.ctx
        let queue = self.queue
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Result, Error>) in
                queue.async {
                    // Cancelled while queued behind an earlier request: skip it.
                    if box.isCancelled {
                        cont.resume(throwing: CancellationError())
                        return
                    }

                    let user = Unmanaged.passRetained(box).toOpaque()
                    defer { Unmanaged<PartialBox>.fromOpaque(user).release() }

                    var p = nllb_translate_default_params()
                    p.max_new_tokens = Int32(maxNewTokens)
                    p.user_data = user
                    // A C function pointer: no captures, everything via user_data.
                    p.on_partial = { textSoFar, user in
                        guard let textSoFar, let user else { return 1 }
                        let box = Unmanaged<PartialBox>.fromOpaque(user).takeUnretainedValue()
                        // Returning 0 aborts at this token boundary. Checking the
                        // flag here closes the window where a cancel arrives
                        // before nllb_translate() has started.
                        if box.isCancelled { return 0 }
                        box.onPartial(String(cString: textSoFar))
                        return 1
                    }

                    var out: UnsafeMutablePointer<CChar>? = nil
                    var nTokens: Int32 = 0
                    var err = [CChar](repeating: 0, count: 1024)
                    let errLen = err.count

                    let rc = text.withCString { t in
                        src.withCString { s in
                            tgt.withCString { g in
                                p.src_lang = s
                                p.tgt_lang = g
                                return nllb_translate(ctx, t, p, &out, &nTokens, &err, errLen)
                            }
                        }
                    }

                    let produced = out.map { String(cString: $0) } ?? ""
                    nllb_string_free(out)

                    switch rc {
                    case Int32(NLLB_TRANSLATE_OK.rawValue):
                        cont.resume(returning: Result(text: produced, tokens: Int(nTokens), aborted: false))
                    case Int32(NLLB_TRANSLATE_ABORTED.rawValue):
                        if box.isCancelled {
                            cont.resume(throwing: CancellationError())
                        } else {
                            cont.resume(returning: Result(text: produced, tokens: Int(nTokens), aborted: true))
                        }
                    default:
                        cont.resume(throwing: NllbError.translate(NllbContext.string(err)))
                    }
                }
            }
        } onCancel: {
            box.cancel()
            nllb_ctx_request_abort(ctx)   // thread-safe by contract (nllb-c.h)
        }
    }

    private static func string(_ buf: [CChar]) -> String {
        buf.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
    }
}

/// Carries the Swift callback and a cancel flag through the C `void *`.
private final class PartialBox: @unchecked Sendable {
    let onPartial: @Sendable (String) -> Void
    private let lock = NSLock()
    private var cancelled = false

    init(_ onPartial: @escaping @Sendable (String) -> Void) { self.onPartial = onPartial }

    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
}
