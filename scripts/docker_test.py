#!/usr/bin/env python3
"""向 PixelPrune + vLLM 服务发送图片请求，验证剪枝后推理是否正常。"""
import base64

from openai import OpenAI

image_path = "assets/doc.jpg"
prompt = "What is the title of this document? Reply in one short sentence."

with open(image_path, "rb") as f:
    image_url = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

client = OpenAI()
resp = client.chat.completions.create(
    model="Qwen3.5-0.8B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ],
    }],
    max_tokens=32768,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={"top_k": 20},
)

msg = resp.choices[0].message
print(f"image     {image_path}")
print(f"prompt    {prompt}")
print(f"finish    {resp.choices[0].finish_reason}")
print(f"content   {msg.content or ''}")
print(f"reasoning {getattr(msg, 'reasoning', None) or ''}")
print(f"usage     {resp.usage}")
