import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// Specialized norm for gemma3
private class RMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
    }
}

// Rotating cache for sliding window attention
public class RotatingKVCache: KVCache {
    let maxSize: Int
    let keep: Int
    
    init(maxSize: Int, keep: Int = 0) {
        self.maxSize = maxSize
        self.keep = keep
        super.init()
    }
    
    override func update(_ keys: MLXArray, _ values: MLXArray) -> (MLXArray, MLXArray) {
        if self.maxSize <= 0 {
            return (keys, values)
        }
        
        let currentLength = keys.dim(2)
        if currentLength > self.maxSize {
            let start = currentLength - self.maxSize
            return (
                keys.sliced([nil, nil, start..., nil]),
                values.sliced([nil, nil, start..., nil])
            )
        }
        return (keys, values)
    }
}

private class Attention: Module {
    let args: Gemma3Configuration
    let scale: Float
    let headDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let isSliding: Bool
    let layerIdx: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: Gemma3Configuration, layerIdx: Int) {
        self.args = args
        self.layerIdx = layerIdx

        let dim = args.hiddenSize
        self.nHeads = args.attentionHeads
        self.nKVHeads = args.kvHeads
        self.repeats = args.attentionHeads / args.kvHeads
        self.headDim = args.headDimensions
        self.isSliding = (layerIdx + 1) % args.slidingWindowPattern != 0

        self.scale = 1.0 / pow(Float(args.queryPreAttnScalar), 0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
        
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        
        let ropeBase = isSliding ? args.ropeLocalBaseFreq : args.ropeGlobalBaseFreq
        self.rope = RoPE(
            dimensions: headDim,
            traditional: args.ropeTraditional,
            base: ropeBase)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)
        
        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        
        queries = qNorm(queries)
        keys = kNorm(keys)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        var currentMask = mask
        if let mask = mask, mask.dim(-1) != keys.dim(-2) {
            currentMask = mask.sliced([nil, nil, (-keys.dim(-2))...])
        }

        queries = queries * self.scale
        
        var scores = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1.0,
            mask: currentMask
        )

        if repeats > 1 {
            scores = scores.reshaped([B, nHeads, L, headDim])
        }
        
        scores = scores.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return wo(scores)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(geluApprox(gate(x)) * up(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Gemma3Configuration, layerIdx: Int) {
        self._attention.wrappedValue = Attention(args, layerIdx: layerIdx)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + postAttentionLayerNorm(r)
        r = mlp(preFeedforwardLayerNorm(h))
        let out = h + postFeedforwardLayerNorm(r)
        return out
    }
}

public class ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    fileprivate let norm: RMSNorm
    private let args: Gemma3Configuration

    public init(_ args: Gemma3Configuration) {
        precondition(args.vocabularySize > 0)
        self.args = args

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0..<args.hiddenLayers).map { layerIdx in
            TransformerBlock(args, layerIdx: layerIdx)
        }
        
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        h = h * pow(Float(args.hiddenSize), 0.5)

        var fullMask: MLXArray? = nil
        var slidingWindowMask: MLXArray? = nil
        
        if cache == nil {
            let j = args.slidingWindowPattern
            fullMask = createAttentionMask(h: h, cache: Array(cache?[j-1...j] ?? []))
            slidingWindowMask = createAttentionMask(h: h, cache: cache)
        }

        for (i, layer) in layers.enumerated() {
            let isSliding = (i % args.slidingWindowPattern) == (args.slidingWindowPattern - 1)
            var currentMask: MLXArray? = nil
            
            if cache == nil {
                currentMask = isSliding ? slidingWindowMask : fullMask
            }
            
            h = layer(h, mask: currentMask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Gemma3Model: Module, UnifiedModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let model: ModelInner
    let args: Gemma3Configuration
    
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: Gemma3Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.args = args
        self.model = ModelInner(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }
    
    public func makeCache() -> [KVCache] {
        return (0..<args.hiddenLayers).map { i in
            if (i % args.slidingWindowPattern) == (args.slidingWindowPattern - 1) {
                return KVCache()
            } else {
                return RotatingKVCache(maxSize: args.slidingWindow, keep: 0)
            }
        }
    }
    
    public func sanitize(_ weights: inout [String: MLXArray]) {
        if !weights.keys.contains("lm_head.weight") {
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        }
        weights = weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
    }
}

public struct Gemma3Configuration: Codable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeGlobalBaseFreq: Float = 1_000_000.0
    var ropeLocalBaseFreq: Float = 10_000.0
    var ropeTraditional: Bool = false
    var queryPreAttnScalar: Float = 256.0
    var slidingWindow: Int = 512
    var slidingWindowPattern: Int = 6

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeGlobalBaseFreq = "rope_global_base_freq"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
    }
}

// MARK: - LoRA

extension Gemma3Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
