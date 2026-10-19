import AVFoundation
import CoreML
import EG2Transcriber
import Foundation

func emit(_ value: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    FileHandle.standardOutput.write(data + Data([10]))
}

do {
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    let model = try EG2Transcriber(folder: root, prefer: .split, units: .cpuAndNeuralEngine, onStage: nil)
    let punctuation = try HeadOnlyPunct(folder: root, computeUnits: .cpuAndNeuralEngine)
    emit(["ready": true, "engine": "eg2-original-1.6-v5", "compute": "cpuAndNeuralEngine"])
    while let line = readLine() {
        do {
            let job = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
            let path = job["path"] as! String
            let url = URL(fileURLWithPath: path)
            let file = try AVAudioFile(forReading: url)
            let duration = Double(file.length) / file.processingFormat.sampleRate
            guard duration <= 3600 else { throw NSError(domain: "STT", code: 400, userInfo: [NSLocalizedDescriptionKey: "Audio exceeds one hour"]) }
            let started = ProcessInfo.processInfo.systemUptime
            let samples = try AudioDecoder.decode16kMono(url, progress: nil)
            let words = try model.transcribe(samples: samples, onWindow: nil, isCancelled: nil)
            let segments: [[String: Any]] = words.map { ["word": $0.text, "start": $0.start, "end": $0.end] }
            emit(["text": TextFormatter.format(words.map(\.text), with: punctuation), "duration": Double(samples.count) / 16000,
                  "seconds": ProcessInfo.processInfo.systemUptime - started, "words": segments])
        } catch { emit(["error": String(describing: error)]) }
    }
} catch {
    emit(["error": String(describing: error)])
    exit(1)
}
