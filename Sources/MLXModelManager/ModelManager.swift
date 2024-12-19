// ModelManager.swift
import Foundation
import Combine
import SwiftUI
import CoreImage
import MLX
import MLXLMCommon
import Tokenizers
import Hub

@MainActor
public class ModelManager: ObservableObject {
    @Published public var progressPercent: Int = 0
    @Published public var output: String = ""
    @Published public var isLoading: Bool = false
    @Published public var isGenerating: Bool = false

    private let modelPath: String
    //private var container: ModelContext?
    public private(set) var container: ModelContext?
    private var temperature: Float = 0.7
    private var topP: Float = 0.9
    private var repetitionPenalty: Float = 1.0

    public init(modelPath: String) {
        self.modelPath = modelPath
    }

    public func setHyperparameters(temperature: Float?, topP: Float?, repetitionPenalty: Float?) {
        if let temp = temperature { self.temperature = temp }
        if let p = topP { self.topP = p }
        if let rp = repetitionPenalty { self.repetitionPenalty = rp }
    }

    public func loadModel() async throws {
        guard !isLoading else { return }
        isLoading = true
        output = "Loading model..."
        do {
            let configuration = ModelConfiguration(id: modelPath)
            let hub = HubApi()
            self.container = try await ModelFactory.shared.load(
                hub: hub,
                configuration: configuration,
                progressHandler: { progress in
                    Task { @MainActor in
                        self.progressPercent = Int(progress.fractionCompleted * 100)
                    }
                }
            )
            print("Debug: loadModel() completed successfully. container: \(String(describing: self.container))")
            output = "Model loaded successfully."
        } catch {
            output += "\nError loading model: \(error.localizedDescription)"
            print("Debug: Error in loadModel(): \(error)")
        }
        isLoading = false
    }

    public func generate(prompt: String, imagePath: String? = nil) async {
        print("Debug: In generate(). container: \(String(describing: container))")
        guard let container else {
            output = "Model not loaded."
            return
        }
        guard !isGenerating else { 
            print("Debug: Already generating, ignoring new request.")
            return 
            }
        
        isGenerating = true
        output = "Generating..."

        do {
            var userInput = UserInput(prompt: .text(prompt))
            if let imagePath = imagePath, !imagePath.isEmpty,
               FileManager.default.fileExists(atPath: imagePath),
               let ciImage = CIImage(contentsOf: URL(fileURLWithPath: imagePath)) {
                userInput.images = [.ciImage(ciImage)]
            } else if let imagePath = imagePath {
                output += "\nWarning: Could not load image at \(imagePath)"
                print("Debug: Could not load image at \(imagePath)")
            }
            // Prepare the LMInput using the model's processor
            print("Debug: Preparing input...")
            let lmInput = try await container.processor.prepare(input: userInput)
            print("Debug: Input prepared.")

            // Set up generation parameters
            let parameters = GenerateParameters(
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty
            )
           
            // This array will hold all generated tokens as we stream
            //var allTokens = [Int]()

            var previouslyDisplayedCount = 0

            print("Debug: Calling generate function")
            // Call the top-level generate function
            let result = try MLXLMCommon.generate(
                input: lmInput,
                parameters: parameters,
                context: container
            ) 
            { tokens in
            let newTokens = tokens[previouslyDisplayedCount..<tokens.count]
            let partialText = container.tokenizer.decode(tokens: Array(newTokens))
            self.output += partialText
            previouslyDisplayedCount = tokens.count
            return .more
            }

            /*{ _ in
                // Return .more to keep generating until EOS or limit is reached
                .more
            }*/
            print("Debug: Generation completed.")

            // Decode the result
            //output = result.output
            self.output = result.output
            print("Debug: Output: \(result.output)")


        } catch {
            output += "\nGeneration error: \(error.localizedDescription)"
            print("Debug: Generation error: \(error)")
        }

        isGenerating = false
    }
}

