# 引入tqdm用于显示进度条
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image
import torch
import os
import re


class ImageRetrievalLocal:
    """本地模型版本的图像检索（已升级并恢复了特征预计算进度条）"""

    def __init__(self, clip_model_path="./model/clip/clip-vit-large-patch14",
                 image_folder="./model/image_lib/val2017",
                 translator_model_path="./model/nllb-200-distilled-600M"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"正在从本地路径 '{clip_model_path}' 加载CLIP模型...")
        self.model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_path)
        print("CLIP模型加载成功。")

        print(f"正在从本地路径 '{translator_model_path}' 加载NLLB翻译模型...")
        device_id = 0 if torch.cuda.is_available() else -1
        self.translator = pipeline(
            'translation',
            model=translator_model_path,
            device=device_id
        )
        print("NLLB翻译模型加载成功。")

        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if
                            f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"图像库包含 {len(self.image_paths)} 张图片。")

        # 调用特征预计算方法
        self.image_features = self._precompute_image_features()

    def _is_chinese(self, text):
        """判断文本是否包含中文字符"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def _translate_zh_to_en(self, text):
        """中文到英文翻译 (使用NLLB模型)"""
        result = self.translator(text, src_lang='zho_Hans', tgt_lang='eng_Latn')
        return result[0]['translation_text']

    def _precompute_image_features(self):
        """
        预计算图像库中所有图像的特征向量，并在控制台显示进度。
        这是一个一次性的、在系统启动时执行的离线计算过程。
        """
        features = []
        with torch.no_grad():
            # --- 核心修改：使用tqdm包装循环以显示进度条 ---
            # desc参数为进度条提供了一个描述性标签
            for path in tqdm(self.image_paths, desc="正在预计算图像库特征"):
                try:
                    image = Image.open(path)
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    image_feature = self.model.get_image_features(**inputs)
                    features.append(image_feature)
                except Exception as e:
                    # 如果某张图片处理失败，打印错误并跳过
                    print(f"\n处理图片失败 '{path}': {e}")

        print("图像库特征计算完成。")
        # 将特征列表拼接成一个张量
        return torch.cat(features)

    def search_images(self, query_text, top_k=5):
        """根据文本检索图像"""
        try:
            query_english = query_text
            if self._is_chinese(query_text):
                print(f"检测到中文输入，正在翻译: '{query_text}'")
                query_english = self._translate_zh_to_en(query_text)
                print(f"翻译结果: '{query_english}'")

            inputs = self.processor(text=query_english, return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

            similarities = torch.nn.functional.cosine_similarity(text_features, self.image_features)
            top_k_indices = torch.topk(similarities, top_k).indices.tolist()

            results = [{
                "image_path": self.image_paths[i],
                "similarity": similarities[i].item()
            } for i in top_k_indices]

            return {
                "success": True,
                "error": None,
                "query_english": query_english,
                "results": results
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query_english": "",
                "results": []
            }