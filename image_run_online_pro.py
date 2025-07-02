import streamlit as st
from image_search_online import ImageRetrievalOnline
from image_text_online import ImageCaptioningOnline
from PIL import Image
import sys

# 使用st.cache_resource确保整个加载过程只在应用启动时执行一次
@st.cache_resource
def load_online_models_with_progress():
    """
    加载所有在线模型，并为每个阶段提供清晰的UI反馈。
    详细的下载进度将显示在运行本程序的控制台中。
    """
    header_placeholder = st.empty()
    header_placeholder.header("系统正在初始化，请稍候...")
    st.info("首次运行时，系统将从Hugging Face Hub下载所需模型。请在您启动本程序的控制台/终端窗口查看详细的下载进度。")

    try:
        # --- 加载图像描述模块 ---
        caption_status = st.empty()
        caption_status.info("阶段 1/2: 正在准备图像描述模块...")
        with st.spinner("正在加载 ViT-GPT2 和 NLLB 翻译模型..."):
# vvvvvv 以下是第 1 处修改 vvvvvv
            image_captioning = ImageCaptioningOnline(
                caption_model_name="nlpconnect/vit-gpt2-image-captioning",
                translator_model_name="facebook/nllb-200-distilled-600M" # <-- 替换为NLLB模型
            )
# ^^^^^^ 第 1 处修改结束 ^^^^^^
        caption_status.success("图像描述模块准备就绪。")

        # --- 加载图像检索模块 ---
        retrieval_status = st.empty()
        retrieval_status.info("阶段 2/2: 正在准备图像检索模块...")
        with st.spinner("正在加载 CLIP 和 NLLB 翻译模型..."):
# vvvvvv 以下是第 2 处修改 vvvvvv
            image_retrieval = ImageRetrievalOnline(
                clip_model_name="openai/clip-vit-large-patch14",
                image_folder="./model/image_lib/val2017",
                translator_model_name="facebook/nllb-200-distilled-600M" # <-- 替换为NLLB模型
            )
# ^^^^^^ 第 2 处修改结束 ^^^^^^
        retrieval_status.success("图像检索模块准备就绪。")
        header_placeholder.empty()
        st.success("系统初始化完成，所有模块已上线！")

        return image_retrieval, image_captioning

    except Exception as e:
        st.error(f"系统初始化失败，请检查网络连接或模型名称是否正确。错误信息: {e}")
        sys.exit()

# --- 主逻辑 (无修改) ---
st.set_page_config(page_title="多模态交互系统 (在线模型版)", layout="wide")
col1, col2 = st.columns([3, 1])

with col1:
    st.title("多模态交互系统(在线模型版)")

with col2:
    st.markdown(
        """
        <style>
        .info-container {
            padding-top: 2.75rem;
            text-align: right;
        }
        .info-container p {
            color: #000000;
            margin-bottom: 0;
            line-height: 1.5;
        }
        </style>
        <div class="info-container">
            <p>姓名：林鑫科<br>学号：1193220223</p>
        </div>
        """,
        unsafe_allow_html=True
    )

image_retrieval_model, image_captioning_model = load_online_models_with_progress()

st.markdown("---")
tab1, tab2 = st.tabs(["🖼️ 图像描述生成", "🔍 文本检索图像"])

with tab1:
    st.header("上传图片，自动生成中英文描述")
    uploaded_file = st.file_uploader("请在此处上传一张图片...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="已上传的图片", use_column_width=True)
        with st.spinner('正在生成描述，请稍候...'):
            result = image_captioning_model.generate_caption(image)
        if result["success"]:
            st.success("描述生成成功！")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("英文描述")
                st.write(result["english"])
            with col2:
                st.subheader("中文描述")
                st.write(result["chinese"])
        else:
            st.error(f"生成描述时出错: {result['error']}")

with tab2:
    st.header("输入文本，从图库中检索相关图片")
    query_text = st.text_input("输入中文或英文描述", placeholder="例如：一只猫躺在沙发上")
    if st.button("开始检索", use_container_width=True):
        if not query_text:
            st.warning("请输入描述以检索图片。")
        else:
            with st.spinner(f"正在检索与 “{query_text}” 相关的图片..."):
                result = image_retrieval_model.search_images(query_text, top_k=5)
            if result["success"]:
                st.success(f"检索完成！(英文查询: '{result['query_english']}')")
                if not result["results"]:
                    st.info(f"在图库中没有找到与“{query_text}”相关的图片。")
                else:
                    image_paths = [item["image_path"] for item in result["results"]]
                    st.image(image_paths, caption=[f"相似度: {item['similarity']:.4f}" for item in result["results"]],
                             width=200)
            else:
                st.error(f"检索时出错: {result['error']}")