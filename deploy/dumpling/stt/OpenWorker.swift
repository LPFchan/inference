import AVFoundation
import Foundation

@main struct Worker {
    static func emit(_ value: [String: Any]) {
        let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        FileHandle.standardOutput.write(data + Data([10]))
    }

    static func main() async {
        do {
            let folder = try ModelFolder(root: URL(fileURLWithPath: CommandLine.arguments[1]))
            let formID = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "gpu-ane"
            guard let form = folder.availableForms.first(where: { $0.id == formID }) else {
                throw TranscribeError.model("No pipeline form \(formID)")
            }
            let model = try await Transcriber.load(folder: folder, form: form)
            emit(["ready": true, "engine": "eg2-open-reader-v2", "compute": formID])
            while let line = readLine() {
                do {
                    let job = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
                    let url = URL(fileURLWithPath: job["path"] as! String)
                    let file = try AVAudioFile(forReading: url)
                    guard Double(file.length) / file.processingFormat.sampleRate <= 3600 else {
                        throw TranscribeError.model("Audio exceeds one hour")
                    }
                    let result = try await model.transcribe(url, overlap: true)
                    emit(["text": result.text, "duration": result.audioSeconds,
                          "seconds": result.totalSeconds, "words": []])
                } catch { emit(["error": String(describing: error)]) }
            }
        } catch {
            emit(["error": String(describing: error)])
            exit(1)
        }
    }
}
