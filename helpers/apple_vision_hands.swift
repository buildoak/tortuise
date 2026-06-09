import AVFoundation
import Foundation
import Vision

final class HandVisionDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let request = VNDetectHumanHandPoseRequest()
    private let targetInterval: TimeInterval
    private var lastEmit = Date.distantPast
    private var sequence: UInt64 = 0

    init(targetFps: Int) {
        self.targetInterval = 1.0 / Double(max(1, min(60, targetFps)))
        super.init()
        self.request.maximumHandCount = 2
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let now = Date()
        if now.timeIntervalSince(lastEmit) < targetInterval {
            return
        }
        lastEmit = now
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

func parseTargetFps() -> Int {
    let args = CommandLine.arguments
    for idx in args.indices {
        if args[idx] == "--fps", idx + 1 < args.count, let value = Int(args[idx + 1]) {
            return max(1, min(60, value))
        }
    }
    return 30
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
    let delegate = HandVisionDelegate(targetFps: parseTargetFps())
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
