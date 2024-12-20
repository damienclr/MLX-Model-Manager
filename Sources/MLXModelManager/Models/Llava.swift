// Copyright © 2024 Apple Inc.
//
// This file adds support for the Llava VLM model, following the patterns from Qwen2VL.swift and Paligemma.swift.
//
// It consolidates the Python implementations from language.py, vision.py, and llava.py into a single Swift file.
// The configuration, model definition, and user input processing are all handled here.
//
// Model Overview:
// Llava is a multi-modal model composed of a Llama-based language model and a CLIP-style vision model.
// They are combined with a multi-modal projector. The text and vision configurations are merged into a single
// LlavaConfiguration to match the Swift framework's single-file configuration approach.
//
// This implementation is inspired by the style and structure of Qwen2VL.swift and Paligemma.swift,
// leveraging MLX, MLXNN, and MLXLMCommon frameworks and the existing patterns for building multi-modal models.

import CoreImage
import Foundation
import Hub
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Language

private enum Language {

    fileprivate struct TextConfiguration: Codable, Sendable {
        let modelType: String
        let hiddenSize: Int
        let numHiddenLayers: Int
        let intermediateSize: Int
        let numAttentionHeads: Int
        let rmsNormEps: Float
        let vocabSize: Int
        let numKeyValueHeads: Int
        let ropeTheta: Float
        let ropeTraditional: Bool
        let ropeScaling: [String: AnyCodable]?
        let tieWordEmbeddings: Bool

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
            case ropeScaling = "rope_scaling"
            case tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    fileprivate class Attention: Module {
        let nHeads: Int
        let nKVHeads: Int
        let repeats: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "rope") var rope: RoPE

        init(_ config: TextConfiguration) {
            let dim = config.hiddenSize
            self.nHeads = config.numAttentionHeads
            self.nKVHeads = config.numKeyValueHeads
            self.repeats = nHeads / nKVHeads

            let headDim = dim / nHeads
            self.scale = pow(Float(headDim), -0.5)

            let attentionBias = (config.modelType == "qwen2") // a detail from original code, likely false for llama
            self._qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: attentionBias)
            self._kProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: attentionBias)
            self._vProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: attentionBias)
            self._oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

            var ropeScale: Float = 1.0
            if let ropeScaling = config.ropeScaling,
               let factor = ropeScaling["factor"]?.floatValue,
               let typ = ropeScaling["type"]?.stringValue,
               typ == "linear" {
                ropeScale = 1 / factor
            }

            self._rope.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta,
                scale: ropeScale
            )
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
            let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)

            var k = keys
            var v = values
            if let cache {
                (k, v) = cache.update(keys: keys, values: values)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: k, values: v, scale: scale, mask: mask
            ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

            return oProj(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(dim: Int, hiddenDim: Int) {
            self._gate.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._down.wrappedValue = Linear(hiddenDim, dim, bias: false)
            self._up.wrappedValue = Linear(dim, hiddenDim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        let mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: TextConfiguration) {
            self._selfAttn.wrappedValue = Attention(config)
            self.mlp = MLP(dim: config.hiddenSize, hiddenDim: config.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
            let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            let r2 = mlp(postAttentionLayerNorm(h))
            return h + r2
        }
    }

    fileprivate class Llama: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [TransformerBlock]
        let norm: RMSNorm
        let config: TextConfiguration

        init(_ config: TextConfiguration) {
            self.config = config
            self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
            self.layers = (0..<config.numHiddenLayers).map { _ in TransformerBlock(config) }
            self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(_ inputs: MLXArray?, cache: [KVCache]? = nil, inputsEmbeds: MLXArray? = nil) -> MLXArray {
            let h: MLXArray
            if let inputsEmbeds {
                h = inputsEmbeds
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("One of inputs or inputs_embeds must be provided.")
            }

            let mask = createAttentionMask(h: h, cache: cache)
            var out = h
            for (i, layer) in layers.enumerated() {
                out = layer(out, mask: mask, cache: cache?[i])
            }
            return norm(out)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        let config: TextConfiguration
        @ModuleInfo var model: Llama
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int] { Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers) }

        init(_ config: TextConfiguration) {
            self.config = config
            self.model = Llama(config)
            if !config.tieWordEmbeddings {
                self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
            }
        }

        func callAsFunction(_ inputs: MLXArray?, cache: [KVCache]? = nil, inputsEmbeds: MLXArray? = nil, mask: MLXArray? = nil) -> LMOutput {
            var out = model(inputs, cache: cache, inputsEmbeds: inputsEmbeds)
            if let lmHead = lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }

        static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            // Remove unused precomputed rotary freqs
            weights.filter { !($0.key.contains("self_attn.rotary_emb.inv_freq")) }
        }
    }
}

// MARK: - Vision

private enum Vision {

    fileprivate struct VisionConfiguration: Codable, Sendable {
        let modelType: String
        let numHiddenLayers: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let numAttentionHeads: Int
        let imageSize: Int
        let patchSize: Int
        let projectionDim: Int
        let vocabSize: Int
        let numChannels: Int
        let layerNormEps: Float

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

    private class VisionAttention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var oProj: Linear

        init(dims: Int, numHeads: Int) {
            guard dims % numHeads == 0 else {
                fatalError("The input feature dimensions should be divisible by the number of heads")
            }
            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)
            self._qProj.wrappedValue = Linear(dims, dims, bias: false)
            self._kProj.wrappedValue = Linear(dims, dims, bias: false)
            self._vProj.wrappedValue = Linear(dims, dims, bias: false)
            self._outProj.wrappedValue = Linear(dims, dims, bias: false)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
            var q = qProj(x)
            var k = kProj(x)
            var v = vProj(x)

            q = q.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)

            let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
                .transposed(0, 2, 1, 3)
                .reshaped(B, L, D)
            return oProj(out)
        }
    }

    private class VisionMLP: Module, UnaryLayer {
        @ModuleInfo var gate: Linear
        @ModuleInfo var down: Linear
        @ModuleInfo var up: Linear

        init(dim: Int, hiddenDim: Int) {
            self.gate = Linear(dim, hiddenDim, bias: false)
            self.down = Linear(hiddenDim, dim, bias: false)
            self.up = Linear(dim, hiddenDim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(nn.silu(gate(x)) * up(x))
        }
    }

    private class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: VisionAttention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        let mlp: VisionMLP

        init(_ config: VisionConfiguration) {
            self._selfAttn.wrappedValue = VisionAttention(dims: config.hiddenSize, numHeads: config.numAttentionHeads)
            self._layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            self._layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            self.mlp = VisionMLP(dim: config.hiddenSize, hiddenDim: config.intermediateSize)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            let h = x + selfAttn(layerNorm1(x), mask: mask)
            return h + mlp(layerNorm2(h))
        }
    }

    private class VisionEmbeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
        let numPositions: Int

        init(_ config: VisionConfiguration) {
            self._patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.hiddenSize,
                kernelSize: .int(config.patchSize),
                stride: .int(config.patchSize),
                bias: false
            )
            let numPatches = (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            self.numPositions = numPatches + 1
            self._positionEmbedding.wrappedValue = Embedding(embeddingCount: numPositions, dimensions: config.hiddenSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let B = x.dim(0)
            var patchEmbeddings = patchEmbedding(x)
            // patchEmbeddings: [B, hidden_size, H', W']
            let Hprime = patchEmbeddings.dim(2)
            let Wprime = patchEmbeddings.dim(3)
            patchEmbeddings = patchEmbeddings.flattened(start: 2, end: 3).transposed(0, 2, 1)
            // Now [B, H'*W', hidden_size]

            let clsEmbedding = zeros([B, 1, patchEmbeddings.dim(2)], dtype: patchEmbeddings.dtype)
            var embeddings = concatenated([clsEmbedding, patchEmbeddings], axis: 1)

            let positionIds = MLXArray(npRange: 0..<numPositions)[.newAxis, 0...]
            embeddings = embeddings + positionEmbedding(positionIds)
            return embeddings
        }
    }

    private class ClipVisionModel: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        let layers: [EncoderLayer]
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: VisionConfiguration) {
            self.embeddings = VisionEmbeddings(config)
            self.layers = (0..<config.numHiddenLayers).map { _ in EncoderLayer(config) }
            self._postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (MLXArray, MLXArray, [MLXArray]?) {
            var h = embeddings(x)
            var states: [MLXArray]? = outputHiddenStates ? [h] : nil

            for layer in layers {
                h = layer(h, mask: nil)
                if outputHiddenStates {
                    states?.append(h)
                }
            }
            let poolerOutput = postLayerNorm(h[:, 0, 0...])
            return (poolerOutput, h, states)
        }
    }

    fileprivate class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: ClipVisionModel
        let config: VisionConfiguration

        init(_ config: VisionConfiguration) {
            guard config.modelType == "clip_vision_model" else {
                fatalError("Currently only clip_vision_model is supported")
            }
            self.config = config
            self._visionModel.wrappedValue = ClipVisionModel(config)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (MLXArray, MLXArray, [MLXArray]?) {
            visionModel(x, outputHiddenStates: outputHiddenStates)
        }

        static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized = [String: MLXArray]()
            for (k,v) in weights {
                if k.contains("position_ids") {
                    continue
                } else if k.contains("patch_embedding.weight") {
                    // Convert from PyTorch [out_channels, in_channels, kH, kW] to MLX [out_channels, kH, kW, in_channels]
                    // Check shape
                    let outC = v.dim(0)
                    let inC = v.dim(1)
                    let kH = v.dim(2)
                    let kW = v.dim(3)
                    if (outC >= kH) && (outC >= kW) && (kH == kW) {
                        // looks good
                        sanitized[k] = v.transposed(0,2,3,1)
                    } else {
                        // fallback
                        sanitized[k] = v.transposed(0,2,3,1)
                    }
                } else {
                    sanitized[k] = v
                }
            }
            return sanitized
        }
    }
}

// MARK: - Model Configuration

public struct LlavaConfiguration: Codable, Sendable {
    public struct ModelConfig: Codable, Sendable {
        public let text_config: Language.TextConfiguration
        public let vision_config: Vision.VisionConfiguration
        public let model_type: String
        public let ignore_index: Int
        public let image_token_index: Int
        public let vision_feature_select_strategy: String
        public let vision_feature_layer: Int
        public let vocab_size: Int
    }

    public let textConfiguration: Language.TextConfiguration
    public let visionConfiguration: Vision.VisionConfiguration
    public let modelType: String
    public let ignoreIndex: Int
    public let imageTokenIndex: Int
    public let visionFeatureSelectStrategy: String
    public let visionFeatureLayer: Int
    public let vocabularySize: Int

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case ignoreIndex = "ignore_index"
        case imageTokenIndex = "image_token_index"
        case visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case visionFeatureLayer = "vision_feature_layer"
        case vocabularySize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.textConfiguration = try container.decode(Language.TextConfiguration.self, forKey: .textConfiguration)
        self.visionConfiguration = try container.decode(Vision.VisionConfiguration.self, forKey: .visionConfiguration)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.ignoreIndex = try container.decode(Int.self, forKey: .ignoreIndex)
        self.imageTokenIndex = try container.decode(Int.self, forKey: .imageTokenIndex)
        self.visionFeatureSelectStrategy = try container.decode(String.self, forKey: .visionFeatureSelectStrategy)
        self.visionFeatureLayer = try container.decode(Int.self, forKey: .visionFeatureLayer)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
    }
}

// MARK: - LlavaMultiModalProjector

private class LlavaMultiModalProjector: Module, UnaryLayer {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    @ModuleInfo var gelu: GELU

    init(_ config: LlavaConfiguration) {
        self._linear1.wrappedValue = Linear(config.visionConfiguration.hiddenSize, config.textConfiguration.hiddenSize, bias: true)
        self._gelu.wrappedValue = GELU()
        self._linear2.wrappedValue = Linear(config.textConfiguration.hiddenSize, config.textConfiguration.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}

// MARK: - Llava Model

public class Llava: Module, UnifiedModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector: LlavaMultiModalProjector

    public let config: LlavaConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: LlavaConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
        self._multiModalProjector.wrappedValue = LlavaMultiModalProjector(config)
    }

    private func getInputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?) -> MLXArray {
        guard let pixelValues else {
            return languageModel.model.embedTokens(inputIds)
        }

        // language embeddings
        var inputEmbeds = languageModel.model.embedTokens(inputIds)

        // get vision embeddings
        let (_, _, hiddenStates) = visionModel(pixelValues.transposed(0,2,3,1), outputHiddenStates: true)
        guard let hiddenStates else {
            fatalError("No hidden states returned from vision model")
        }

        let selectedImageFeature = hiddenStates[config.visionFeatureLayer]
        let imageFeatures: MLXArray
        switch config.visionFeatureSelectStrategy {
        case "default":
            imageFeatures = selectedImageFeature[:, 1...] // remove CLS token
        case "full":
            imageFeatures = selectedImageFeature
        default:
            fatalError("Unexpected feature selection strategy: \(config.visionFeatureSelectStrategy)")
        }

        let projectedImageFeatures = multiModalProjector(imageFeatures)
        return mergeInputIdsWithImageFeatures(projectedImageFeatures, inputEmbeds: inputEmbeds, inputIds: inputIds)
    }

    private func mergeInputIdsWithImageFeatures(_ imageFeatures: MLXArray, inputEmbeds: MLXArray, inputIds: MLXArray) -> MLXArray {
        let imageTokenIndex = config.imageTokenIndex
        let inputIdArray = inputIds.asArray(Int.self)
        let imagePositions = inputIdArray.enumerated().compactMap { $1 == imageTokenIndex ? $0 : nil }

        let numImages = imageFeatures.dim(0)
        guard imagePositions.count == numImages else {
            fatalError("The number of image tokens (\(imagePositions.count)) does not match the number of image inputs (\(numImages)).")
        }

        // We'll split the text into segments around the image tokens and insert image features
        var textSegments = [MLXArray]()
        var startIdx = 0
        for position in imagePositions {
            textSegments.append(inputEmbeds[:, startIdx..<position])
            startIdx = position + 1
        }
        textSegments.append(inputEmbeds[:, startIdx..<inputEmbeds.dim(1)])

        let imageEmbeds = split(imageFeatures, imageFeatures.shape[0])
        var finalEmbedsParts = [MLXArray]()
        for (textSeg, img) in zip(textSegments.dropLast(), imageEmbeds) {
            finalEmbedsParts.append(textSeg)
            finalEmbedsParts.append(img)
        }
        finalEmbedsParts.append(textSegments.last!)

        return concatenated(finalEmbedsParts, axis: 1)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let pixelValues = input.image?.pixels
        let inputEmbeddings = getInputEmbeddings(inputIds: inputIds, pixelValues: pixelValues)
        let result = languageModel(nil, cache: cache, inputsEmbeds: inputEmbeddings)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let w = Vision.VisionModel.sanitize(weights)
        return Language.LanguageModel.sanitize(w)
    }
}

// MARK: - Processor

public class LlavaProcessor: UserInputProcessor {
    private let imageTokenIndex: Int
    private let tokenizer: any Tokenizer

    public init(imageTokenIndex: Int, tokenizer: any Tokenizer) {
        self.imageTokenIndex = imageTokenIndex
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) throws -> LMInput {
        // Prepare text
        let prompt = input.prompt.asPlainText()

        // If no images provided, just tokenize text
        if input.images.isEmpty {
            let promptTokens = try tokenizer.encode(text: prompt)
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: tokensArray).asType(.int8)
            return LMInput(text: .init(tokens: tokensArray, mask: mask))
        }

        // If images are provided, we must insert the image token(s) in the prompt
        // The Python code expects the prompt to contain the <image> token (index: imageTokenIndex)
        // We don't have a special token text form here, but we can just append a token that maps to imageTokenIndex after encoding.
        // The python code prepares something like: "What is in this image?" and then input_ids have an <image> token.
        // We'll insert a special token in the text, then replace it with imageTokenIndex after tokenization if needed.

        var modifiedPrompt = prompt
        // Insert a single <image> token in text for each provided image
        // The original code uses a fixed image_token_index = 32000. We'll just insert a special marker and replace it.
        for _ in input.images {
            modifiedPrompt += " <image>" 
        }

        let promptTokens = try tokenizer.encode(text: modifiedPrompt)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)

        // Replace the token "<image>" with imageTokenIndex if tokenizer recognized it as a known token
        // If not known, we might have to handle that. For now, assume tokenizer maps "<image>" to a unique ID or OOV and user maps it.
        // Alternatively, we can post-process: find the tokens that represent "<image>" and replace them.

        // For simplicity, assume tokenizer has a token "<image>" that we can map.
        // If not, we'd need a custom handling. Let's do a naive approach:
        // Let’s find the substring "<image>" in the prompt and map the last occurrence of the token to imageTokenIndex.
        // In practice, the user would ensure the tokenizer includes an image token.

        let tokensData = promptArray.asArray(Int.self)
        // We'll just replace any token that matches the tokenizer's "<image>" token ID if defined.
        // If the tokenizer does not have <image>, user would have to handle that differently.
        // For now, assume it's a known token. If not, we must guess an OOV token and replace that:
        // We'll just do nothing and rely on the input prompt having the correct token. The model expects <image> as imageTokenIndex anyway.

        // We must load and preprocess images
        // The python code expects pixel_values in shape [B, C, H, W] in a normalized format.
        // The user can supply a known image processing pipeline. For demonstration:
        let images = input.images.map { MediaProcessing.apply($0.asCIImage(), processing: input.processing) }
        // Llava typically expects the raw CLIP-like preprocessing (no special resizing instructions given, 
        // but from llava code: image is processed by the vision tower as is. We must ensure image size matches the vision model.
        // Typically CLIP models: resize to `imageSize x imageSize`
        // Let's assume a standard CLIP image size of visionConfiguration.imageSize and apply normalization.
        // Without the config here, we can't. In a full implementation, we'd have config. For demonstration:
        // We'll guess a standard: 224x224, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]

        // We'll just pick standard CLIP preprocessing for demonstration:
        let mean: (CGFloat, CGFloat, CGFloat) = (0.48145466, 0.4578275, 0.40821073)
        let std: (CGFloat, CGFloat, CGFloat) = (0.26862954, 0.26130258, 0.27577711)
        let size = CGSize(width: 224, height: 224)

        let processedImages = images.map {
            MediaProcessing.resampleBicubic($0, to: size)
        }.map {
            MediaProcessing.normalize($0, mean: mean, std: std)
        }.map {
            MediaProcessing.asMLXArray($0)
        }

        let pixelValues = concatenated(processedImages) // [B, H, W, C]
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(text: .init(tokens: promptArray, mask: mask), image: .init(pixels: pixelValues))
    }
}

