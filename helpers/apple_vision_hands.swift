import AVFoundation
import Foundation
import Vision

final class HandVisionDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let request = VNDetectHumanHandPoseRequest()
    private let targetSampleInterval: TimeInterval
    private let previewEnabled: Bool
    private let previewWidth: Int
    private let previewHeight: Int
    private let previewInterval: TimeInterval
    private var lastSampleEmit = Date.distantPast
    private var lastPreviewEmit = Date.distantPast
    private var sequence: UInt64 = 0
    private var previewSequence: UInt64 = 0

    init(targetFps: Int, previewEnabled: Bool, previewWidth: Int, previewHeight: Int, previewFps: Int) {
        self.targetSampleInterval = 1.0 / Double(max(1, min(60, targetFps)))
        self.previewEnabled = previewEnabled
        self.previewWidth = max(16, min(192, previewWidth))
        self.previewHeight = max(9, min(108, previewHeight))
        self.previewInterval = 1.0 / Double(max(1, min(30, previewFps)))
        super.init()
        self.request.maximumHandCount = 2
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let now = Date()
        if previewEnabled && now.timeIntervalSince(lastPreviewEmit) >= previewInterval {
            lastPreviewEmit = now
            previewSequence += 1
            if let payload = previewPayload(sampleBuffer: sampleBuffer) {
                print("preview \(previewSequence) \(previewWidth) \(previewHeight) \(payload)")
                fflush(stdout)
            }
        }

        if now.timeIntervalSince(lastSampleEmit) < targetSampleInterval {
            return
        }
        lastSampleEmit = now
        sequence += 1

        let started = Date()
        let handler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("error vision_perform")
            fflush(stdout)
            return
        }

        let detectMs = Date().timeIntervalSince(started) * 1000.0
        let observations = request.results ?? []
        var parts: [String] = []
        for (idx, observation) in observations.enumerated() {
            guard let summary = summarizeHand(observation: observation, index: idx) else {
                continue
            }
            parts.append(summary)
        }

        if parts.isEmpty {
            print(String(format: "sample %llu %.3f", sequence, detectMs))
        } else {
            print(String(format: "sample %llu %.3f %@", sequence, detectMs, parts.joined(separator: " ")))
        }
        fflush(stdout)
    }

    private func previewPayload(sampleBuffer: CMSampleBuffer) -> String? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }

        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let source = baseAddress.assumingMemoryBound(to: UInt8.self)
        var rgb = [UInt8](repeating: 0, count: previewWidth * previewHeight * 3)

        for y in 0..<previewHeight {
            let sy = min(sourceHeight - 1, y * sourceHeight / previewHeight)
            for x in 0..<previewWidth {
                let sx = min(sourceWidth - 1, sourceWidth - 1 - (x * sourceWidth / previewWidth))
                let srcIdx = sy * bytesPerRow + sx * 4
                let dstIdx = (y * previewWidth + x) * 3
                rgb[dstIdx] = source[srcIdx + 2]
                rgb[dstIdx + 1] = source[srcIdx + 1]
                rgb[dstIdx + 2] = source[srcIdx]
            }
        }

        return Data(rgb).base64EncodedString()
    }

    private func summarizeHand(observation: VNHumanHandPoseObservation, index: Int) -> String? {
        guard let points = try? observation.recognizedPoints(.all) else {
            return nil
        }

        let reliable = points.values.filter { $0.confidence >= 0.25 }
        guard !reliable.isEmpty else {
            return nil
        }

        let center = reliable.reduce(CGPoint.zero) { acc, point in
            CGPoint(x: acc.x + point.location.x, y: acc.y + point.location.y)
        }
        let inv = 1.0 / CGFloat(reliable.count)
        let centerX = center.x * inv
        let centerY = center.y * inv
        let avgConfidence = reliable.reduce(Float(0.0)) { $0 + $1.confidence } / Float(reliable.count)

        let pinch = pinchScore(points: points)
        // Mirror X so front-camera movement feels like the browser prototype.
        return String(
            format: "%d,%.5f,%.5f,%.5f,%.5f",
            index,
            1.0 - Double(centerX),
            Double(centerY),
            Double(pinch),
            Double(avgConfidence)
        )
    }

    private func pinchScore(points: [VNHumanHandPoseObservation.JointName: VNRecognizedPoint]) -> Float {
        guard
            let thumb = points[.thumbTip],
            let index = points[.indexTip],
            thumb.confidence >= 0.25,
            index.confidence >= 0.25
        else {
            return 0.0
        }
        let dx = thumb.location.x - index.location.x
        let dy = thumb.location.y - index.location.y
        let distance = sqrt(dx * dx + dy * dy)
        let score = 1.0 - Float(distance / 0.18)
        return max(0.0, min(1.0, score))
    }
}

struct HelperArgs {
    var targetFps: Int = 30
    var previewEnabled: Bool = false
    var previewWidth: Int = 64
    var previewHeight: Int = 36
    var previewFps: Int = 8
}

func parseHelperArgs() -> HelperArgs {
    let args = CommandLine.arguments
    var parsed = HelperArgs()
    for idx in args.indices {
        if args[idx] == "--fps", idx + 1 < args.count, let value = Int(args[idx + 1]) {
            parsed.targetFps = max(1, min(60, value))
        } else if args[idx] == "--preview" {
            parsed.previewEnabled = true
        } else if args[idx] == "--preview-width", idx + 1 < args.count, let value = Int(args[idx + 1]) {
            parsed.previewWidth = max(16, min(192, value))
        } else if args[idx] == "--preview-height", idx + 1 < args.count, let value = Int(args[idx + 1]) {
            parsed.previewHeight = max(9, min(108, value))
        } else if args[idx] == "--preview-fps", idx + 1 < args.count, let value = Int(args[idx + 1]) {
            parsed.previewFps = max(1, min(30, value))
        }
    }
    return parsed
}

func requestCameraAccessIfNeeded() -> Bool {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
        return true
    case .notDetermined:
        let semaphore = DispatchSemaphore(value: 0)
        var granted = false
        AVCaptureDevice.requestAccess(for: .video) { ok in
            granted = ok
            semaphore.signal()
        }
        semaphore.wait()
        return granted
    case .denied, .restricted:
        return false
    @unknown default:
        return false
    }
}

func main() {
    let args = parseHelperArgs()
    guard requestCameraAccessIfNeeded() else {
        print("error camera_denied")
        fflush(stdout)
        exit(2)
    }

    let session = AVCaptureSession()
    session.sessionPreset = .vga640x480

    let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
        ?? AVCaptureDevice.default(for: .video)
    guard let device else {
        print("error camera_unavailable")
        fflush(stdout)
        exit(3)
    }

    do {
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            print("error camera_input")
            fflush(stdout)
            exit(4)
        }
        session.addInput(input)
    } catch {
        print("error camera_input")
        fflush(stdout)
        exit(4)
    }

    let output = AVCaptureVideoDataOutput()
    output.alwaysDiscardsLateVideoFrames = true
    output.videoSettings = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ]
    let delegate = HandVisionDelegate(
        targetFps: args.targetFps,
        previewEnabled: args.previewEnabled,
        previewWidth: args.previewWidth,
        previewHeight: args.previewHeight,
        previewFps: args.previewFps
    )
    output.setSampleBufferDelegate(delegate, queue: DispatchQueue(label: "tortuise.apple-vision-hands"))
    guard session.canAddOutput(output) else {
        print("error camera_output")
        fflush(stdout)
        exit(5)
    }
    session.addOutput(output)

    if let connection = output.connection(with: .video), connection.isVideoMirroringSupported {
        connection.isVideoMirrored = true
    }

    print("status apple_vision_ready")
    fflush(stdout)
    session.startRunning()
    RunLoop.main.run()
}

main()
