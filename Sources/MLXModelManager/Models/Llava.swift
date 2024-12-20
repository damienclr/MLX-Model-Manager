// Llava.swift

import Foundation
import CoreImage
import MLX
import MLXFast
import MLXNN
import MLXLMCommon
import Tokenizers

// MARK: - Configurations

public struct LlavaConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let numKeyValueHeads: Int
    public let ropeTheta: Float
    public let ropeTraditional: Bool
    public let tieWordEmbeddings: Bool

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decode(String.self, forKey: .modelType)
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11008
        self.numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32000
        self.numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? self.numAttentionHeads
        self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        self.ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case numKeyValueHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case tieWordEmbeddings = "tie_word_embeddings"
    }
}

// We assume we have TextConfig and VisionConfig types already from previous code
// If not, define them similarly or match them to your actual fields.

public struct VisionConfig: Codable, Sendable {
    public let modelType: String
    public let numHiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let imageSize: Int
    public let patchSize: Int
    public let projectionDim: Int
    public let vocabSize: Int
    public let numChannels: Int
    public let layerNormEps: Float

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decode(String.self, forKey: .modelType)
        self.numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        self.numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 336
        self.patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        self.projectionDim = try c.decodeIfPresent(Int.self, forKey: .projectionDim) ?? 768
        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32000
        self.numChannels = try c.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        self.layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numHiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case projectionDim = "projection_dim"
        case vocabSize = "vocab_size"
        case numChannels = "num_channels"
        case layerNormEps = "layer_norm_eps"
    }
}

// TextConfig used by LanguageModel
public struct TextConfig: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let numKeyValueHeads: Int
    public let ropeTheta: Float
    public let ropeTraditional: Bool
    public let tieWordEmbeddings: Bool
}

// ModelConfig used by Llava
public struct ModelConfig: Codable, Sendable {
    public let textConfig: TextConfig
    public let visionConfig: VisionConfig
    public let modelType: String
    public let ignoreIndex: Int
    public let imageTokenIndex: Int
    public let visionFeatureSelectStrategy: String
    public let visionFeatureLayer: Int
    public let vocabSize: Int
}

// MARK: - Llava Model

public class Llava: Module, UnifiedModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") var visionModel: VisionModel
    @ModuleInfo(key: "language_model") var languageModel: LanguageModel
    @ModuleInfo(key: "multi_modal_projector") var projector: MultiModalProjector

    public let config: ModelConfig

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    // Change init to accept LlavaConfiguration
    public init(_ llavaConfig: LlavaConfiguration) {
        // Convert LlavaConfiguration to TextConfig, VisionConfig, and ModelConfig

        let textConfig = TextConfig(
            modelType: llavaConfig.modelType,
            hiddenSize: llavaConfig.hiddenSize,
            numHiddenLayers: llavaConfig.numHiddenLayers,
            intermediateSize: llavaConfig.intermediateSize,
            numAttentionHeads: llavaConfig.numAttentionHeads,
            rmsNormEps: llavaConfig.rmsNormEps,
            vocabSize: llavaConfig.vocabSize,
            numKeyValueHeads: llavaConfig.numKeyValueHeads,
            ropeTheta: llavaConfig.ropeTheta,
            ropeTraditional: llavaConfig.ropeTraditional,
            tieWordEmbeddings: llavaConfig.tieWordEmbeddings
        )

        // For now, use dummy VisionConfig or decode from `config.json` if needed.
        // You must have vision config info from config.json. If it's not part of LlavaConfiguration,
        // either add it or provide defaults:
        let visionConfig = VisionConfig(
            modelType: "clip_vision_model",
            numHiddenLayers: 24,
            hiddenSize: 1024,
            intermediateSize: 4096,
            numAttentionHeads: 16,
            imageSize: 336,
            patchSize: 14,
            projectionDim: 768,
            vocabSize: 32000,
            numChannels: 3,
            layerNormEps: 1e-5
        )

        // Construct ModelConfig with defaults or read from full config file:
        // If your final code reads from config.json, you might already have a ModelConfig. For now:
        let modelConfig = ModelConfig(
            textConfig: textConfig,
            visionConfig: visionConfig,
            modelType: llavaConfig.modelType,
            ignoreIndex: -100,
            imageTokenIndex: 32000,
            visionFeatureSelectStrategy: "default",
            visionFeatureLayer: -2,
            vocabSize: llavaConfig.vocabSize
        )

        self.config = modelConfig

        // Initialize submodules using modelConfig as before
        self._visionModel.wrappedValue = VisionModel(modelConfig.visionConfig)
        self._languageModel.wrappedValue = LanguageModel(modelConfig.textConfig)
        self._multi_modal_projector.wrappedValue = MultiModalProjector(modelConfig)
    }

    public func loraLinearLayers() -> LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?) -> MLXArray {
        guard let pixelValues else {
            return languageModel.model.embedTokens(inputIds)
        }

        let textEmbeds = languageModel.model.embedTokens(inputIds)
        let (poolerOutput, lastHidden, hiddenStates) = visionModel(pixelValues, outputHiddenStates: true)
        let selectedImageFeature = hiddenStates?.last ?? lastHidden
        let imageFeatures = projector(selectedImageFeature)

        var result = textEmbeds
        let imageTokenId = config.imageTokenIndex
        let inputIdsArray = inputIds.asArray(Int.self)
        for (i, token) in inputIdsArray.enumerated() {
            if token == imageTokenId {
                result[0..., i...(i+imageFeatures.dim(1)-1), 0...] = imageFeatures
            }
        }

        return result
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        let pixels = input.image?.pixels
        let inputEmbedding = inputEmbeddings(input.text.tokens, pixelValues: pixels)
        let result = languageModel(nil, cache: cache, inputsEmbeds: inputEmbedding)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = VisionModel.sanitize(weights)
        w = LanguageModel.sanitize(w)
        return w
    }
}

