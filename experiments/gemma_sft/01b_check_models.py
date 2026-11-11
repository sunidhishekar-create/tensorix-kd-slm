import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print("Gemma models visible to this key:")
for m in client.models.list():
    if "gemma" in m.name.lower():
        print("  ", m.name)

for model in ["gemma-4-31b-it", "gemma-4-26b-a4b-it", "gemini-3.8-flash"]:
    try:
        r = client.models.generate_content(model=model, contents="Say hello in five words.")
        print(model, "OK:", r.text.strip())
    except Exception as e:
        print(model, "FAILED:", str(e)[:150])