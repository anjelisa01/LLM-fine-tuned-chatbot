import zipfile
import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, ".")

def ask_model(prompt, model=model, tokenizer=tokenizer, max_new_tokens=20):
    """
    Generate an answer from the fine-tuned model based on a custom prompt.
    Args:
        prompt (str): Your custom question or instruction.
        model: Your fine-tuned Hugging Face model.
        tokenizer: The tokenizer used with the model.
        max_new_tokens (int): Max number of tokens to generate (default 20).
    Returns:
        str: Cleaned assistant response.
    """
    # Format prompt like the training data
    full_prompt = f"### Human: {prompt}\n### Assistant:"
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    # Generate response
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )
    # Decode and clean
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the assistant answer
    if "### Assistant:" in decoded:
        answer = decoded.split("### Assistant:")[1].strip()
        # Cut off hallucinated continuation (like another ### block or file paths)
        for stop_token in ["### Human:", "###", "\n#", "\n##"]:
            if stop_token in answer:
                answer = answer.split(stop_token)[0].strip()
    return answer


# Set up Gradio 
iface = gr.Interface(fn=ask_model, 
                     inputs=gr.Textbox(label="Prompt", placeholder="Enter text here..."), 
                     outputs="text",
                     title="Simple chatbot using Fine-tuned GPT-Neo with LoRA",
                     description="This app uses a fine-tuned GPT-Neo model with LoRA for text generation.")

# Launch the Gradio app
iface.launch()
