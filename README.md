# MLX Model Manager

MLX Model Manager provides a unified interface for loading and running both Large Language Models (LLMs) and Vision-Language Models (VLMs) on Apple Silicon. It simplifies the model download, initialization, and generation process, so you can focus on integrating AI capabilities into your applications.

<video width="600" controls>
  <source src="https://github.com/kunal732/MLX-Model-Manager/releases/download/v0.0.1/Paligemma2-4k.mov" type="video/mp4">
  Your browser does not support the video tag.
</video>

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

struct VLMContentView: View {
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

## Attribution

This is built on top of [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples). Originally developed by [davidkoski](https://github.com/davidkoski), [awni](https://github.com/awni), and the MLX team, with additional contributions from the community. Special thanks to Prince Canuma, who created [Blaizzy](https://github.com/Blaizzy), later ported to Swift by the MLX team.

MLX Model Manager unifies how you can interact with both LLMs and VLMs, streamlining development and integration efforts.

