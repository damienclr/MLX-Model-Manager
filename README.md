# MLX Model Manager

MLX Model Manager provides a unified interface for loading and running both Large Language Models (LLMs) and Vision-Language Models (VLMs) on Apple Silicon. It simplifies the model download, initialization, and generation process, so you can focus on integrating AI capabilities into your applications.


https://github.com/user-attachments/assets/80da0647-c428-4cf6-8e47-43d2dd659291


## Installation

📦 Use Swift Package Manager to integrate MLX Model Manager into your project.

1. Open Xcode and go to **File > Add Packages...**.
2. Enter the repository URL for `mlx-model-manager`.
3. Add `MLXModelManager` to your target.
4. Use `import MLXModelManager` in your Swift project.

## Usage

```
func loadAndGenerate() async {
    let manager = ModelManager(modelPath: "mlx-community/phi-2-hf-4bit-mlx")
    do {
        // Load the model
        try await manager.loadModel()
        print("Model loaded successfully.")
        
        // Generate text
        await manager.generate(prompt: "What is the capital of France?")
        print("Generated Output: \(manager.output)")
    } catch {
        print("Error: \(error)")
    }
}
```


### Example: Loading an LLM

```
import SwiftUI
import MLXModelManager

struct ContentView: View {
    @StateObject var manager = ModelManager(modelPath:"mlx-community/phi-2-hf-4bit-mlx")

    var body: some View {
        VStack(spacing: 20) {
            Text("🤖 LLM Example").font(.headline)
            Button("Load & Generate") {
                Task {
                    try await manager.loadModel()
                    await manager.generate(prompt: "What is the capital of France?")
                }
            }
            ScrollView {
                Text(manager.output).padding()
            }
        }
        .padding()
    }
}
```

### Example: Loading a VLM

```
import SwiftUI
import MLXModelManager

struct ContentView: View {
    @StateObject var manager = ModelManager(modelPath: "mlx-community/paligemma2-3b-ft-docci-448-8bit")

    var body: some View {
        VStack(spacing: 20) {
            Text("🖼️ VLM Example").font(.headline)
            Button("Load & Describe Image") {
                Task {
                    try await manager.loadModel()
                    await manager.generate(
                        prompt: "Describe the image in English.",
                        imagePath: "/path/to/your/image.png"
                    )
                }
            }
            ScrollView {
                Text(manager.output).padding()
            }
        }
        .padding()
    }
}
```

## Parameters

⚙️ You can adjust various parameters before generating text:

-  `temperature`: Controls randomness. Lower values = more deterministic output.
-  `topP`: Nucleus sampling threshold. Consider only tokens within cumulative probability `topP`.
-  `repetitionPenalty`: Penalizes repeated tokens to encourage diversity.
-  `maxTokens` (optional): Limits the maximum number of tokens to generate.

Set parameters before calling `generate`:

```
manager.maxTokens = 200
manager.setHyperparameters(temperature: 0.7, topP: 0.9, repetitionPenalty: 1.0)
try await manager.loadModel()
await manager.generate(prompt: "Explain quantum mechanics simply.")
```

## Supported Models

You can replace the `modelPath` parameter in your Swift code with any of the IDs listed below. All models support Language modality, and the last two also support Vision.

| ModelPath                                      | Language | Vision |
|------------------------------------------------|:--------:|:------:|
| mlx-community/SmolLM-135M-Instruct-4bit       | ✅       |        |
| mlx-community/Mistral-Nemo-Instruct-2407-4bit | ✅       |        |
| mlx-community/Mistral-7B-Instruct-v0.3-4bit   | ✅       |        |
| mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX | ✅    |        |
| mlx-community/phi-2-hf-4bit-mlx               | ✅       |        |
| mlx-community/Phi-3.5-mini-instruct-4bit      | ✅       |        |
| mlx-community/Phi-3.5-MoE-instruct-4bit       | ✅       |        |
| mlx-community/quantized-gemma-2b-it           | ✅       |        |
| mlx-community/gemma-2-9b-it-4bit              | ✅       |        |
| mlx-community/gemma-2-2b-it-4bit              | ✅       |        |
| mlx-community/Qwen1.5-0.5B-Chat-4bit          | ✅       |        |
| mlx-community/OpenELM-270M-Instruct           | ✅       |        |
| mlx-community/Meta-Llama-3.1-8B-Instruct-4bit | ✅       |        |
| mlx-community/Meta-Llama-3-8B-Instruct-4bit   | ✅       |        |
| mlx-community/Llama-3.2-1B-Instruct-4bit      | ✅       |        |
| mlx-community/Llama-3.2-3B-Instruct-4bit      | ✅       |        |
| mlx-community/paligemma-3b-mix-448-8bit       | ✅       | ✅     |
| mlx-community/Qwen2-VL-2B-Instruct-4bit       | ✅       | ✅     |
| mlx-community/paligemma2-3b-ft-docci-448-8bit | ✅       | ✅     |

## Attribution

This is built on top of [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples). Originally developed by [davidkoski](https://github.com/davidkoski), [awni](https://github.com/awni), and the MLX team, with additional contributions from the community. Special thanks to [Prince Canuma](https://github.com/Blaizzy), who created [MLX-VLM](https://github.com/Blaizzy/mlx-vlm), later ported to Swift by the MLX team.

MLX Model Manager unifies how you can interact with both LLMs and VLMs, streamlining development and integration efforts.

