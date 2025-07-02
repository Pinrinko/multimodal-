from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm


class ImageRetrievalOnline:
    """
    在线模型版本的文本检索图像系统 (最终性能优化版 - 启用FP16)
    """

    def __init__(self, clip_model_name="openai/clip-vit-large-patch14",
                 image_folder="./model/image_lib/val2017",
                 translator_model_name="facebook/nllb-200-distilled-600M"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vvvvvv 以下是最终、最关键的性能优化 vvvvvv
        # 在加载模型时，明确指定使用FP16半精度，这将极大地提升在兼容GPU上的推理速度
        print("正在从Hugging Face Hub加载CLIP模型 (强制使用FP16半精度)...")
        self.clip_model = CLIPModel.from_pretrained(
            clip_model_name,
            torch_dtype=torch.float16
        ).to(self.device)
# ^^^^^^ 优化部分结束 ^^^^^^

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        device_id = 0 if torch.cuda.is_available() else -1
        self.translator = pipeline("translation", model=translator_model_name, device=device_id)

        self.image_folder = image_folder
        self.image_features = None
        self.valid_paths = []
        self._preprocess_local_image_library()

        if self.valid_paths:
            print(f"本地图像库加载并处理完成，包含 {len(self.valid_paths)} 张有效图片。")

    def _preprocess_local_image_library(self):
        """
        预处理本地图像库，提取所有图像特征 (采用单张处理模式)。
        """
        if not os.path.exists(self.image_folder):
            print(f"错误: 本地图像库路径不存在: {self.image_folder}")
            return

        self.valid_paths = [os.path.join(self.image_folder, f)
                            for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

        all_features_gpu = []
        print("开始预计算本地图像库特征 (采用单张处理和FP16优化)...")
        with torch.no_grad():
            for path in tqdm(self.valid_paths, desc="在控制台提取图像特征"):
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                    # 由于模型已是FP16，这里的输入可能也需要适配，但通常库会自动处理
                    # 如果遇到类型不匹配错误，可尝试 inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device, dtype=torch.float16)
                    features = self.clip_model.get_image_features(**inputs)
                    all_features_gpu.append(features)
                except Exception as e:
                    print(f"\n处理图片失败 '{path}': {e}，已跳过。")

        if all_features_gpu:
            stacked_features = torch.cat(all_features_gpu, dim=0)
            self.image_features = F.normalize(stacked_features, dim=-1).cpu()
            print("图像库特征计算完成。")

    def _is_chinese(self, text):
        """判断文本是否包含中文字符"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def _translate_zh_to_en(self, text):
        """使用NLLB模型将中文翻译为英文"""
        return self.translator(text, src_lang='zho_Hans', tgt_lang='eng_Latn')[0]['translation_text']

    def search_images(self, query_text, top_k=5):
        """根据文本查询检索相似图像"""
        try:
            if self.image_features is None:
                return {"success": False, "error": "本地图像库为空或特征加载失败", "results": []}

            query_en = query_text
            if self._is_chinese(query_text):
                query_en = self._translate_zh_to_en(query_text)

            inputs = self.clip_processor(text=[query_en], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                text_features = F.normalize(text_features, dim=-1)

            similarity = (text_features.cpu() @ self.image_features.T).squeeze(0)

            top_k = min(top_k, len(self.valid_paths))
            top_indices = similarity.topk(top_k).indices

            results = [{"image_path": self.valid_paths[idx.item()], "similarity": similarity[idx].item()} for idx in
                       top_indices]

            return {"success": True, "error": None, "query_english": query_en, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e), "results": []}
