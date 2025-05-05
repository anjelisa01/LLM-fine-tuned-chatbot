# LLM-fine-tuned-chatbot (GPT-Neo Fine-tuned with LoRA)

[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/anjel01/simple-chatbot)

A  chatbot fine-tuned on [GPT-Neo 1.3B] using Low-Rank Adaptation (LoRA). Built for lightweight interaction and experimentation with conversational models.

## Features
- Fine-tuned on custom dialogue dataset using LoRA adapters
- Lightweight deployment with Gradio UI
- Hosted on Hugging Face Spaces

## Model
- Base model: EleutherAI/gpt-neo-1.3B
- Fine-tuning: LoRA via PEFT
- Trained on: generated conversation text. example: "### Human: What is the capital of France?\n### Assistant: Paris."
- The fine-tuned GPT-Neo model is saved directly in the Hugging Face Space and loaded at runtime.
- Not available as a standalone model on the Hugging Face Hub, but included in the deployed app.
