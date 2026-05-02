from openai import OpenAI
import time

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

system_prompt = "You are a networking expert with deep knowledge of TCP, UDP, QUIC, and congestion control algorithms including Reno, CUBIC, and BBR. You have published extensively on transport protocol design, flow control mechanisms, and network performance optimization. Your expertise covers both theoretical foundations and practical deployment considerations. Please provide detailed technical answers."

questions = [
    "Explain TCP congestion control.",
    "Compare TCP to QUIC.",
]

for i, q in enumerate(questions):
    print(f"\n{'='*60}")
    print(f"REQUEST {i+1}")
    print(f"{'='*60}")
    print(f"Q: {q}\n")

    t = time.time()
    r = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ],
        # max_tokens=100,
    )
    elapsed = time.time() - t

    print(f"A: {r.choices[0].message.content}\n")
    print(f"[{elapsed:.3f}s | prompt_tokens={r.usage.prompt_tokens} | completion_tokens={r.usage.completion_tokens}]")