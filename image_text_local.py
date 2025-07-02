from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
from PIL import Image
import torch

class ImageCaptioningLocal:
    """本地模型版本的图像描述（已升级为从本地路径加载NLLB翻译模型）"""

    def __init__(self, caption_model_path="./model/vit-gpt2-image-captioning",
                 translator_model_path="./model/nllb-200-distilled-600M"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"正在从本地路径 '{caption_model_path}' 加载图像描述模型...")
        self.caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_path).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(caption_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(caption_model_path)

        print(f"正在从本地路径 '{translator_model_path}' 加载NLLB翻译模型...")
        device_id = 0 if torch.cuda.is_available() else -1
        self.translator = pipeline(
            'translation',
            model=translator_model_path,
            device=device_id
        )
        print("NLLB翻译模型加载成功。")

        print(f"所有本地模型已加载，使用设备: {self.device}")

    def translate_en_to_zh(self, text):
        """英文到中文翻译 (使用NLLB模型)"""
        result = self.translator(text, src_lang='eng_Latn', tgt_lang='zho_Hans')
        return result[0]['translation_text']

    def generate_caption(self, image_path, max_length=16, do_sample=True,
                         top_k=50, top_p=0.95, temperature=1.0):
        """生成图像描述"""
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path.convert("RGB")

            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

            with torch.no_grad():
                output_ids = self.caption_model.generate(
                    pixel_values,
                    max_length=max_length,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature
                )

            caption_en = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            caption_zh = self.translate_en_to_zh(caption_en)

            return {
                "english": caption_en,
                "chinese": caption_zh,
                "success": True,
                "error": None
            }

        except Exception as e:
            return {
                "english": "",
                "chinese": "",
                "success": False,
                "error": str(e)
            }