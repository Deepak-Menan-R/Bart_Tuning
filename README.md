---
title: BartTune
emoji: 🏃
colorFrom: red
colorTo: red
sdk: docker
pinned: false
short_description: User Requirements to Use case (Fine tuned Bart-BASE)
---

# BartTune: Domain-Specific User Story Generation for College Management Systems

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://mrcodder-barttune.hf.space/)
[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)

> **A Final Year Academic Project**: Domain-specific fine-tuning of BART-BASE for automated user story generation in college management systems.

## 🎯 Problem Statement

In software development, especially for domain-specific applications like college management systems, converting user requirements into well-structured user stories is a time-consuming and expertise-dependent task. This process requires:

- Understanding domain-specific terminology and workflows
- Structuring requirements into standardized formats (title, actors, goals, preconditions, flows, etc.)
- Ensuring consistency across multiple user stories
- Manual effort that slows down the requirements engineering phase

**BartTune** addresses this challenge by leveraging fine-tuned transformer models to automatically generate comprehensive user stories from simple use case descriptions, specifically optimized for college management system scenarios.

## 🔬 Research Overview

This project explores the effectiveness of fine-tuning BART (Bidirectional and Auto-Regressive Transformers) models across different training epochs to optimize the quality of generated user stories. We trained and evaluated 8 different model variants to identify the optimal training configuration.

### Key Contributions
- **Domain-Specific Fine-Tuning**: Specialized training on college management system use cases
- **Epoch Analysis**: Comparative study of model performance across 1, 3, 5, 8, 10, 12, 15, and 20 epochs
- **Production-Ready Application**: Deployed web interface with authentication and real-time generation
- **Comprehensive Evaluation**: Multi-metric assessment using BLEU, ROUGE, and BERTScore

## 📊 Model Performance

| Epoch | Loss  | BLEU  | ROUGE-1 | ROUGE-L | BERTScore | Runtime (s) | Samples/sec |
|-------|-------|-------|---------|---------|-----------|-------------|-------------|
| **1** | 1.795 | 0.362 | 0.647   | 0.533   | 0.900     | 171.49      | 0.268       |
| **3** | 1.175 | 0.484 | 0.737   | 0.640   | 0.923     | 143.16      | 0.321       |
| **5** | 0.898 | 0.528 | 0.767   | 0.681   | 0.930     | 144.73      | 0.318       |
| **8** | 0.647 | 0.596 | 0.806   | 0.741   | 0.942     | 140.28      | 0.328       |
| **10**| 0.537 | 0.640 | 0.833   | 0.779   | 0.950     | 155.08      | 0.297       |
| **12**| 0.449 | 0.676 | 0.852   | 0.808   | 0.956     | 143.50      | 0.321       |
| **15**| 0.343 | 0.736 | 0.881   | 0.849   | 0.964     | 144.23      | 0.319       |
| **20**| 0.217 | 0.828 | 0.922   | 0.907   | 0.976     | 139.76      | 0.329       |

### 🏆 Best Performing Model: Epoch 20
- **BLEU Score**: 0.828 (82.8% n-gram precision)
- **ROUGE-1**: 0.922 (92.2% unigram overlap)
- **ROUGE-L**: 0.907 (90.7% longest common subsequence)
- **BERTScore**: 0.976 (97.6% semantic similarity)

## 💡 Example Usage

### Input (Use Case)
```
Students able to register themself in the system
```

### Output (Generated User Story - Epoch 20 Model)
```
Title: Register for Registrations
PrimaryActor: User (Student)
Goal: To register for registration.
Preconditions: The user is logged in and on the registration page.
Trigger: The registration window opens.
MainFlow: 
  - The system displays a form for registering.
  - The user fills in their registration details and submits them.
ErrorHandling: 
  - If registration fails, the system notifies the user.
EdgeCases: 
  - Multiple registration attempts: System displays an error message.
Postconditions: Registration is successfully completed.
Assumptions: Registrations are open for a limited, predefined period.
```

## 🏗️ System Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Browser (User)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Web Application                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Authentication Layer                                 │   │
│  │  - Email/Password Auth                               │   │
│  │  - Google OAuth 2.0                                  │   │
│  │  - Session Management                                │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  API Endpoints                                       │   │
│  │  - /signup, /login, /logout                         │   │
│  │  - /chat (Protected Route)                          │   │
│  │  - /generate (Model Inference)                      │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────┬───────────────────┬──────────────────────┘
                   │                   │
                   ▼                   ▼
        ┌──────────────────┐  ┌────────────────────────────┐
        │  MongoDB Atlas   │  │  BART Models (Preloaded)   │
        │  User Database   │  │  - Epoch 1, 3, 5, 8        │
        │  - Credentials   │  │  - Epoch 10, 12, 15, 20    │
        │  - Sessions      │  │  - GPU/CPU Inference       │
        └──────────────────┘  └────────────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────────┐
                              │  Hugging Face Hub        │
                              │  Model Repository        │
                              │  MrCodder/BartBase...    │
                              └──────────────────────────┘
```

### Technical Stack

#### Backend
- **Framework**: FastAPI (Python 3.9)
- **Authentication**: 
  - Authlib (OAuth 2.0)
  - Session-based authentication
  - Google OAuth integration
- **Database**: MongoDB Atlas
- **ML Framework**: PyTorch + Hugging Face Transformers

#### Model Architecture
- **Base Model**: BART-BASE (facebook/bart-base)
- **Fine-tuning Approach**: Supervised sequence-to-sequence learning
- **Inference**: GPU-accelerated (CUDA) with CPU fallback
- **Deployment**: Model preloading at startup for zero-latency inference

#### Frontend
- **Templates**: Jinja2
- **Styling**: Bootstrap 5
- **JavaScript**: Vanilla JS for dynamic interactions

#### Deployment
- **Platform**: Hugging Face Spaces
- **Containerization**: Docker
- **Port**: 7860 (Standard HF Spaces port)
- **Server**: Uvicorn ASGI

### Key Features

1. **🔐 Secure Authentication**
   - Email/Password registration and login
   - Google OAuth 2.0 integration
   - Session-based access control
   - Secure password hashing (SHA-256)

2. **🤖 Multi-Model Support**
   - 8 fine-tuned models available (1-20 epochs)
   - Real-time model selection
   - GPU-accelerated inference
   - Preloaded models for instant response

3. **💬 Interactive Chat Interface**
   - Real-time user story generation
   - Chat-based conversation flow
   - Responsive design for all devices
   - User-friendly epoch selection

4. **📈 Comprehensive Results Dashboard**
   - Detailed evaluation metrics
   - Performance comparison across epochs
   - Interactive tables and visualizations

## 🚀 Live Demo

**Access the application**: [https://mrcodder-barttune.hf.space/](https://mrcodder-barttune.hf.space/)

### Demo Features
- Try different fine-tuned models (1-20 epochs)
- Compare output quality across training iterations
- Test with college management system use cases
- View comprehensive evaluation metrics

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- MongoDB Atlas account (or local MongoDB)
- Google Cloud Console project (for OAuth)
- GPU (optional, but recommended for faster inference)

### Local Development

1. **Clone the Repository**
```bash
git clone <your-repository-url>
cd BartTune
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment Configuration**

Create a `.env` file in the root directory:
```env
# MongoDB Configuration
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/

# Google OAuth
GOOGLE_CLIENT_ID=your_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_client_secret

# Session Security
SECRET_KEY=your_super_secret_key_here

# Environment
ENVIRONMENT=development  # or 'production'
```

5. **Run the Application**
```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

6. **Access Locally**
```
http://localhost:7860
```

### Docker Deployment

```bash
# Build the image
docker build -t barttune .

# Run the container
docker run -p 7860:7860 --env-file .env barttune
```

## 📖 Usage Guide

1. **Sign Up/Login**: Create an account or use Google OAuth
2. **Navigate to Chat**: Access the chat interface after authentication
3. **Select Model**: Choose an epoch (1-20) for generation
4. **Enter Use Case**: Type your college management use case
5. **Generate**: Receive a structured user story instantly
6. **Compare**: Try different epochs to compare output quality

## 🧪 Model Details

### Training Dataset
- **Domain**: College Management System
- **Format**: Use Case → User Story pairs
- **Structure**: Includes title, actors, goals, preconditions, flows, error handling, edge cases, postconditions, and assumptions

### Fine-Tuning Configuration
- **Base Model**: `facebook/bart-base`
- **Training Epochs**: 1, 3, 5, 8, 10, 12, 15, 20
- **Optimization**: AdamW optimizer
- **Generation Parameters**:
  - Max Length: 150 tokens
  - Num Beams: 5 (beam search)
  - Early Stopping: Enabled

### Model Repository
All fine-tuned models are available on Hugging Face:
- `MrCodder/BartBaseFineTuningModel1`
- `MrCodder/BartBaseFineTuningModel3`
- ... (up to epoch 20)

## 👥 Team

**Amrita Vishwa Vidyapeetham - Final Year Project**

| Name | Roll Number |
|------|-------------|
| Deepak Menan R | CB.EN.U4CSE21014 |
| Suruthi Lavanya A | CB.EN.U4CSE21262 |
| Dharsan S | CB.EN.U4CSE21219 |
| Deepthika R | CB.EN.U4CSE21217 |


## 🔮 Future Enhancements

- [ ] Expand to other domains (healthcare, banking, e-commerce)
- [ ] Support for multilingual user story generation
- [ ] Integration with project management tools (Jira, Trello)
- [ ] API endpoints for third-party integration
- [ ] Advanced analytics and reporting features
- [ ] User feedback loop for continuous improvement

## 📜 License

This is an academic project developed as part of final year curriculum at Amrita Vishwa Vidyapeetham.

## 🙏 Acknowledgments

- Facebook AI for BART-BASE model
- Hugging Face for model hosting and deployment platform
- Amrita Vishwa Vidyapeetham for academic guidance and support

---

**Built with ❤️ by Team BartTune | Amrita Vishwa Vidyapeetham | 2025**
