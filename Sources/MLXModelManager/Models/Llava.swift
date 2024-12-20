here is llava.swift:

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
    public struct TextConfiguration: Codable, Sendable {
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

    // Additional fields from python code (clip_vision_model or siglip_vision_model, etc.)
    // Just set defaults if missing
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

public struct ModelConfig: Codable, Sendable {
    public let textConfig: TextConfig
    public let visionConfig: VisionConfig
    public let modelType: String
    public let ignoreIndex: Int
    public let imageTokenIndex: Int
    public let visionFeatureSelectStrategy: String
    public let visionFeatureLayer: Int
    public let vocabSize: Int

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        self.textConfig = try TextConfig(from: decoder)
        self.visionConfig = try VisionConfig(from: decoder)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "llama"
        self.ignoreIndex = try c.decodeIfPresent(Int.self, forKey: .ignoreIndex) ?? -100
        self.imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 32000
        self.visionFeatureSelectStrategy = try c.decodeIfPresent(String.self, forKey: .visionFeatureSelectStrategy) ?? "default"
        self.visionFeatureLayer = try c.decodeIfPresent(Int.self, forKey: .visionFeatureLayer) ?? -2
        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32000
    }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case ignoreIndex = "ignore_index"
        case imageTokenIndex = "image_token_index"
        case visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case visionFeatureLayer = "vision_feature_layer"
        case vocabSize = "vocab_size"
    }
}

}
// MARK: - LanguageModel

// Based on language.py
// We'll implement a Llama-like model similar to Qwen2VL and LLama from qwen2_vl code.

private class RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
    }
}

// Minimal Llama-like block
private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen2VLDecoderLayerAttention
    let mlp: Qwen2VLDecoderLayerMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: TextConfig) {
        self._attention.wrappedValue = Qwen2VLDecoderLayerAttention(config)
        self.mlp = Qwen2VLDecoderLayerMLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

// We'll define minimal classes for attention and MLP based on language code:
private class Qwen2VLDecoderLayerAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(_ config: TextConfig) {
        let dim = config.hiddenSize
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.headDim = dim / heads
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = wq(x)
        var k = wk(x)
        var v = wv(x)

        q = q.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

private class Qwen2VLDecoderLayerMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private class LlamaModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [TransformerBlock]
    let norm: RMSNorm

    init(_ config: TextConfig) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

        self.layers = (0..<config.numHiddenLayers).map { _ in
            TransformerBlock(config)
        }

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil, inputsEmbeds: MLXArray? = nil) -> MLXArray {
        var h = inputsEmbeds ?? embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache)

        for (layer, c) in zip(layers, cache ?? Array(repeating: nil, count: layers.count)) {
            h = layer(h, mask: mask, cache: c)
        }

        return norm(h)
    }
}

private class LanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo var model: LlamaModel
    @ModuleInfo var lmHead: Linear?

    let kvHeads: [Int]

    init(_ config: TextConfig) {
        self.model = LlamaModel(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
    }

    func callAsFunction(_ inputs: MLXArray?, cache: [KVCache]? = nil, inputsEmbeds: MLXArray? = nil) -> LMOutput {
        var out = model(inputs ?? MLXArray([]), cache: cache, inputsEmbeds: inputsEmbeds)
        if let lmHead = lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return LMOutput(logits: out)
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}

// MARK: - VisionModel

// Based on vision.py and the logic from llava.py example
// We'll assume a simpler approach and note that additional complexity may be needed.

private class VisionModel: Module {
    // For brevity, we'll assume a Clip-like vision model
    // The Python code is complex, involving PatchEmbed, Encoder, etc.
    // Here, we just show a simplified structure.

    @ModuleInfo(key: "vision_model") var visionModelCore: ClipVisionModel

    init(_ config: VisionConfig) {
        // For simplicity, just a single model type
        self._visionModelCore.wrappedValue = ClipVisionModel(config)
    }

    func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (MLXArray, MLXArray, [MLXArray]?) {
        visionModelCore(x, outputHiddenStates: outputHiddenStates)
    }

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (k, v) in weights {
            if k.contains("position_id") { continue }
            // handle patch_embedding weight shape if needed
            sanitized[k] = v
        }
        return sanitized
    }
}

// Placeholder ClipVisionModel to match call signatures:
private class ClipVisionModel: Module {
    let config: VisionConfig

    init(_ config: VisionConfig) {
        self.config = config
    }

    func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (MLXArray, MLXArray, [MLXArray]?) {
        // return (poolerOutput, lastHiddenState, hiddenStates?)
        return (x, x, outputHiddenStates ? [x] : nil)
    }
}

// MARK: - Combined Model (LLaVA style)

public class MultiModalProjector: Module, UnaryLayer {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    let gelu = GELU()

    init(_ config: ModelConfig) {
        self.linear1 = Linear(config.visionConfig.hiddenSize, config.textConfig.hiddenSize, bias: true)
        self.linear2 = Linear(config.textConfig.hiddenSize, config.textConfig.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}

public class Llava: Module, UnifiedModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") var visionModel: VisionModel
    @ModuleInfo(key: "language_model") var languageModel: LanguageModel
    @ModuleInfo(key: "multi_modal_projector") var projector: MultiModalProjector

    public let config: ModelConfig

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: ModelConfig) {
        self.config = config
        self._visionModel.wrappedValue = VisionModel(config.visionConfig)
        self._languageModel.wrappedValue = LanguageModel(config.textConfig)
        self._multi_modal_projector.wrappedValue = MultiModalProjector(config)
    }

    public func loraLinearLayers() -> LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?) -> MLXArray {
        guard let pixelValues else {
            return languageModel.model.embedTokens(inputIds)
        }

        // Get text embeddings
        let textEmbeds = languageModel.model.embedTokens(inputIds)

        // Obtain vision features (placeholder: just return pixelValues as if processed)
        // In real code, run pixelValues through visionModel, select features and run projector.
        let (poolerOutput, lastHidden, hiddenStates) = visionModel(pixelValues, outputHiddenStates: true)
        let selectedImageFeature = hiddenStates?.last ?? lastHidden
        let imageFeatures = projector(selectedImageFeature)

        // Merge image features into textEmbeds at the image token positions:
        var result = textEmbeds
        let imageTokenId = config.imageTokenIndex
        let inputIdsArray = inputIds.asArray(Int.self)
        for (i, token) in inputIdsArray.enumerated() {
            if token == imageTokenId {
                // Insert image features here
                // Assuming one image token per example:
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
        // Combine vision and language sanitization
        var w = VisionModel.sanitize(weights)
        w = LanguageModel.sanitize(w)
        return w
    }
}


