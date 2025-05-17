import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import streamlit as st
import asyncio

MODEL_DIR = "E:\\AIML\\Qwen3-1.7B"  # Path to the model directory

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head"]
)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map=0
)

async def get_llama_response(job_role):
    """Generate roadmap using Qwen3-1.7B asynchronously."""

    # Construct the chat-style prompt with thinking mode enabled
    prompt = (
        "You are a career advisor. Generate a structured learning roadmap for a beginner aspiring to be a {Job_role}. "
        "The response should include the following sections:\n"
        "1. Programming Languages\n"
        "2. Frameworks and Libraries\n"
        "3. Tools and Platforms\n"
        "4. Suggested Projects\n\n"
        "Ensure clarity and conciseness. Avoid markdown formatting. "
        "Start the response with '### Career Roadmap:' and end with '##'.\n\n"
        "### Career Roadmap:\n"
    ).format(Job_role=job_role)

    # Construct the input message for Qwen3-1.7B
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Apply the chat template with thinking enabled
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Prepare model input
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    # Generate response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,  # Adjust token limit as required
        temperature=0.05,
        repetition_penalty=1.1
    )

    # Extract generated output
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Parse thinking content and main response
    try:
        # Identify the 'thinking' token (151668)
        think_index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        think_index = 0

    # thinking_content = tokenizer.decode(output_ids[:think_index], skip_special_tokens=True).strip()
    response_content = tokenizer.decode(output_ids[think_index:], skip_special_tokens=True).strip()

    return response_content

# Streamlit App
st.title("Career Roadmap Generator")
st.subheader("Select a job role to generate a learning roadmap")

# Dropdown for Job Role Selection
job_roles = [
    'Frontend Developer', 'Backend Developer', 'Data Engineer', 'Software Developer',
    'Data Scientist', 'Machine Learning Engineer', 'DevOps Engineer', 'Cloud Engineer',
    'Cybersecurity Analyst', 'Database Administrator', 'Mobile App Developer', 'Full Stack Developer',
    'Data Analyst', 'System Architect', 'Network Engineer', 'Business Intelligence Analyst',
    'Blockchain Developer', 'AI Engineer', 'IT Support Specialist', 'Product Manager',
    'QA Engineer', 'Technical Writer', 'Solutions Architect', 'Site Reliability Engineer (SRE)',
    'UX/UI Designer', 'IT Project Manager', 'Big Data Engineer'
]

selected_role = st.selectbox("Select Job Role:", job_roles, key="job_role_select")

if st.button("Generate Roadmap", key="generate_button"):
    with st.spinner("Generating roadmap..."):
        output_placeholder = st.empty()

        # Run the async function using asyncio
        output_content = asyncio.run(get_llama_response(selected_role))
        output_placeholder.text_area("Roadmap for " + selected_role, output_content, height=300)
