# CareerRoadmap
LLM based career roadmap generator.

Make sure to clone the repo before using Qwen if not using hugging face:
source-https://huggingface.co/Qwen/Qwen3-1.7B

Enable thinking is disabled if want to enable make 
```python
enable_thinking=True
```
and uncomment
```python
thinking_content = tokenizer.decode(output_ids[:think_index], skip_special_tokens=True).strip()
```

Download LLama 2:
source-https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

Run app using:
```python
streamlit run app.py
