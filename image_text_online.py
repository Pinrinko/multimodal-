from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
from PIL import Image
import torch

class ImageCaptioningOnline:
    """在线模型版本的图像描述生成 (已升级为使用NLLB翻译模型)"""

# vvvvvv 以下是第 1 处修改 vvvvvv
    def __init__(self, caption_model_name="nlpconnect/vit-gpt2-image-captioning",
                 translator_model_name="facebook/nllb-200-distilled-600M"): # <-- 更新默认模型名称
# ^^^^^^ 第 1 处修改结束 ^^^^^^
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"正在从Hugging Face Hub加载图像描述模型: {caption_model_name}...")
        self.caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_name).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(caption_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(caption_model_name)
        print("图像描述模型加载成功。")

        print(f"正在从Hugging Face Hub加载NLLB翻译模型: {translator_model_name}...")
        device_id = 0 if torch.cuda.is_available() else -1
# vvvvvv 以下是第 2 处修改 vvvvvv
        # NLLB使用通用的'translation'任务
        self.translator = pipeline(
            "translation",
            model=translator_model_name,
            device=device_id
        )
# ^^^^^^ 第 2 处修改结束 ^^^^^^
        print("NLLB翻译模型加载成功。")

        print(f"在线描述生成模块已加载，使用设备: {self.device}")

# vvvvvv 以下是第 3 处修改 vvvvvv
    def _translate_en_to_zh(self, text):
        """使用NLLB模型将英文翻译为中文"""
        # NLLB需要明确指定源语言和目标语言代码
        return self.translator(text, src_lang='eng_Latn', tgt_lang='zho_Hans')[0]['translation_text']
# ^^^^^^ 第 3 处修改结束 ^^^^^^

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
                    pixel_values, max_length=max_length, do_sample=do_sample,
                    top_k=top_k, top_p=top_p, temperature=temperature
                )

            caption_en = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
# vvvvvv 以下是第 4 处修改 vvvvvv
            # 调用新的翻译方法
            caption_zh = self._translate_en_to_zh(caption_en)
# ^^^^^^ 第 4 处修改结束 ^^^^^^

            return {
                "english": caption_en, "chinese": caption_zh,
                "success": True, "error": None
            }
        except Exception as e:
            return {
                "english": "", "chinese": "",
                "success": False, "error": str(e)
            }