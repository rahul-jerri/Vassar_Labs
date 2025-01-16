 
from transformers import pipeline
 
 
generator = pipeline("text-generation", model="gpt2")
 
 
def generate_text(prompt, max_length, num_return_sequences, temperature, top_p, do_sample, repetition_penalty):
    try:
        # Generate text based on the prompt and parameters
        outputs = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        return [output['generated_text'] for output in outputs]
    except Exception as e:
        return f"Error: {e}"
 
prompt = input("Enter your prompt")
print("Generating text...")

results = generate_text(
    prompt,
    max_length=10,
    num_return_sequences=2,
    temperature=0.7,        # Control randomness (lower = more focused)
    top_p=0.9,              # Use nucleus sampling
    do_sample=True,         # Enable sampling (rather than greedy decoding)
    repetition_penalty=1.2, # Penalize repeated phrases
)
 
# Print the generated text
for i, text in enumerate(results, 1):
    print(f"Generated Text {i}:\n{text}\n")
 
