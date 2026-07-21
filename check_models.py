import google.generativeai as genai
import os

# APIキーをここに一時的にペーストしてください（確認が終わったら消してください）
API_KEY = ""

genai.configure(api_key=API_KEY)

print("利用可能なモデル一覧:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"名前: {m.name}")