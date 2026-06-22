import AppKit
import CoreImage
import os.log
import ScreenCaptureKit

private let lensLog = OSLog(subsystem: "com.zachlatta.freeflow", category: "LiquidGlass")

/// TEMP debug: os_log entries for this app aren't visible via `log show` on this
/// machine, so mirror lens diagnostics to a file while we stabilise the lens.
func lensDebug(_ message: String) {
    let line = "\(Date()) \(message)\n"
    let url = URL(fileURLWithPath: "/tmp/lens-debug.log")
    if let handle = try? FileHandle(forWritingTo: url) {
        handle.seekToEndOfFile()
        handle.write(Data(line.utf8))
        try? handle.close()
    } else {
        try? Data(line.utf8).write(to: url)
    }
}

/// Live "liquid glass" backdrop for the bottom dictation pill.
///
/// macOS's native glass (NSGlassEffectView) cannot refract other apps' windows —
/// at pill sizes it renders as plain frost. This view does the refraction
/// manually: it samples the screen region behind the pill and bends it through a
/// capsule lens (CIGlassLozenge) with saturation + blur for the glass material.
///
/// Sampling uses one-shot SCScreenshotManager captures polled ~20fps instead of
/// an SCStream: a persistent stream makes macOS pin its screen-capture indicator
/// chrome (rounded box + chevron control) around the capturing app's window,
/// which looked like a mystery "square" around the pill. One-shot captures
/// don't create a session, so there's no system chrome.
///
/// Requires Screen Recording permission; without it the view stays transparent
/// and the frost base underneath remains visible as the fallback.
final class LiquidGlassBackdropView: NSView {
    private var pollTask: Task<Void, Never>?
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    /// Extra margin captured around the pill so edge refraction has source pixels.
    private let capturePadding: CGFloat = 24
    private var activePillFrame: NSRect = .zero
    private var hasDeliveredFrame = false
    /// Fired on the main queue when the first lens frame is rendered — used to
    /// hold the pill's fade-in until there's real glass to show (no white flash).
    var onFirstFrame: (() -> Void)?

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.masksToBounds = true
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    // Re-assert the capsule clip on every layout pass — AppKit can rebuild the
    // backing layer.
    override func layout() {
        super.layout()
        layer?.masksToBounds = true
        layer?.cornerRadius = bounds.height / 2
    }

    func start(pillFrameOnScreen: NSRect, screen: NSScreen, excludingWindowNumber windowNumber: Int) {
        guard pillFrameOnScreen != activePillFrame || pollTask == nil else { return }
        activePillFrame = pillFrameOnScreen
        hasDeliveredFrame = false
        pollTask?.cancel()
        pollTask = nil

        guard CGPreflightScreenCaptureAccess() else {
            lensDebug("NOT GRANTED — requested, frost fallback")
            // One-time system prompt; user must grant + relaunch for TCC to apply.
            CGRequestScreenCaptureAccess()
            return
        }
        lensDebug("permission OK, starting poll capture \(pillFrameOnScreen)")

        layer?.cornerRadius = pillFrameOnScreen.height / 2
        let scale = screen.backingScaleFactor
        let padded = pillFrameOnScreen.insetBy(dx: -capturePadding, dy: -capturePadding)
        // sourceRect for SCK is in display-local points with a top-left origin.
        let sourceRect = CGRect(
            x: padded.minX - screen.frame.minX,
            y: screen.frame.maxY - padded.maxY,
            width: padded.width,
            height: padded.height
        )
        let pillSize = pillFrameOnScreen.size
        let pad = capturePadding
        guard let displayID = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID else { return }
        guard #available(macOS 14.0, *) else {
            lensDebug("macOS < 14 — no one-shot capture API, frost fallback")
            return
        }

        pollTask = Task { [weak self] in
            do {
                let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
                guard let display = content.displays.first(where: { $0.displayID == displayID }) else {
                    lensDebug("display \(displayID) not found")
                    return
                }
                let ownWindows = content.windows.filter { $0.windowID == CGWindowID(windowNumber) }
                let filter = SCContentFilter(display: display, excludingWindows: ownWindows)

                let config = SCStreamConfiguration()
                config.sourceRect = sourceRect
                config.width = Int(sourceRect.width * scale)
                config.height = Int(sourceRect.height * scale)
                config.showsCursor = false

                // ONE capture per pill appearance, not a poll: sustained capture
                // makes macOS pin its screen-capture indicator chrome (box +
                // chevron) to the app's window. A single screenshot — like ⌘⇧3 —
                // doesn't get badged. The glass is a frozen refraction of the
                // content at show-time, which is visually identical unless the
                // screen scrolls mid-dictation.
                var attempts = 0
                while !Task.isCancelled, attempts < 5 {
                    attempts += 1
                    if let cgFrame = try? await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config) {
                        lensDebug("one-shot frame \(cgFrame.width)x\(cgFrame.height) (attempt \(attempts))")
                        self?.render(frame: cgFrame, pillSize: pillSize, padding: pad, scale: scale)
                        break
                    }
                    try? await Task.sleep(nanoseconds: 80_000_000)
                }
            } catch {
                lensDebug("poll setup FAILED: \(error)")
            }
        }
    }

    private func render(frame cgFrame: CGImage, pillSize: CGSize, padding: CGFloat, scale: CGFloat) {
        var image = CIImage(cgImage: cgFrame)
        let pad = padding * scale
        let pillW = pillSize.width * scale
        let pillH = pillSize.height * scale
        let lensRadius = pillH / 2

        // Glass material: saturation lift + a whisper of dimming (keeps the pill
        // legible over plain white) + blur.
        if let saturate = CIFilter(name: "CIColorControls") {
            saturate.setValue(image, forKey: kCIInputImageKey)
            saturate.setValue(1.35, forKey: kCIInputSaturationKey)
            saturate.setValue(-0.05, forKey: kCIInputBrightnessKey)
            image = saturate.outputImage ?? image
        }
        if let blur = CIFilter(name: "CIGaussianBlur") {
            blur.setValue(image, forKey: kCIInputImageKey)
            blur.setValue(6.0, forKey: kCIInputRadiusKey)
            image = blur.outputImage ?? image
        }

        // The lens: a capsule-shaped refraction spanning the pill.
        if let lens = CIFilter(name: "CIGlassLozenge") {
            lens.setValue(image, forKey: kCIInputImageKey)
            lens.setValue(CIVector(x: pad + lensRadius, y: pad + pillH / 2), forKey: "inputPoint0")
            lens.setValue(CIVector(x: pad + pillW - lensRadius, y: pad + pillH / 2), forKey: "inputPoint1")
            lens.setValue(lensRadius, forKey: "inputRadius")
            lens.setValue(1.6, forKey: "inputRefraction")
            image = lens.outputImage ?? image
        }

        let pillRect = CGRect(x: pad, y: pad, width: pillW, height: pillH)
        guard let cgImage = ciContext.createCGImage(image.cropped(to: pillRect), from: pillRect) else { return }
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.layer?.contents = cgImage
            if !self.hasDeliveredFrame {
                self.hasDeliveredFrame = true
                self.onFirstFrame?()
            }
        }
    }

    /// Stop capturing but keep the last rendered frame on screen — used during
    /// the dismiss fade so the pill doesn't flash while fading out.
    func freeze() {
        pollTask?.cancel()
        pollTask = nil
    }

    func stop() {
        activePillFrame = .zero
        hasDeliveredFrame = false
        pollTask?.cancel()
        pollTask = nil
        layer?.contents = nil
    }
}
