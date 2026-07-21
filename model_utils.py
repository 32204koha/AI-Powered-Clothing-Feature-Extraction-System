import torch
import json
import google.generativeai as genai
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# 名前をconfigと完全に合わせました
from config import VLM_MODEL_NAME, LLM_MODEL_NAME, LLM_PROMPT_TEMPLATE

class FashionAI:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(LLM_MODEL_NAME)
        
        self.vlm = VisionEncoderDecoderModel.from_pretrained(VLM_MODEL_NAME)
        self.processor = ViTImageProcessor.from_pretrained(VLM_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(VLM_MODEL_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vlm.to(self.device)

    def generate_caption(self, image):
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        output_ids = self.vlm.generate(pixel_values, max_length=100, num_beams=4)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # ジェミニに画像データを渡して属性抽出
    def analyze_with_gemini(self, image, metadata):
        # 1. メタデータをプロンプトに埋め込む
        prompt = LLM_PROMPT_TEMPLATE.format(meta=json.dumps(metadata, ensure_ascii=False))
        
        # 2. Geminiに「プロンプト(テキスト)」と「画像データ」を両方セットで渡す
        response = self.llm.generate_content([prompt, image])
        
        return response.text