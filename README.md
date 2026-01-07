# PhysiCast | Spatial Reasoning Engine

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://physicast.onrender.com/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/MelvynAv/physicast)

**PhysiCast** est un moteur d'IA physique développé pour démontrer des capacités avancées en perception visuelle et en raisonnement spatial.

# PhysiCast: Spatial Semantic Interpreter

**PhysiCast** is a Physical AI prototype designed for real time scene interpretation and spatial reasoning. By bridging computer vision and localized large language models, the system converts raw environmental data into actionable **Spatial Knowledge Graphs**.

### Core Capabilities
* **Scene Interpretation**: Leverages **YOLOv8** to extract precise 2D object coordinates and semantic labels.
* **Spatial Reasoning**: Utilizes **Llama 3.2** (via Ollama) to infer physical relationships, proximity, and interaction zones.
* **Edge AI Architecture**: Built with **FastAPI** to ensure low latency and data privacy through local execution.

### Project Architecture
1. **Input**: User uploads a physical scene image via the web dashboard.
2. **Vision Engine**: YOLOv8 detects objects and generates spatial bounding box data.
3. **Reasoning Engine**: A customized prompt feeds coordinates into a local LLM for context aware interpretation.
4. **Output**: A semantic description of the physical space for robotic or digital interaction.

### Technical Stack
* **Language**: Python 3.12
* **Framework**: FastAPI, Uvicorn
* **Computer Vision**: Ultralytics YOLOv8
* **Inference Engine**: Ollama (Llama 3.2)
