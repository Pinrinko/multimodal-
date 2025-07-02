import streamlit as st
from image_search_local import ImageRetrievalLocal
from image_text_local import ImageCaptioningLocal
from PIL import Image
import sys
#  streamlit run D:\桌面\小学期实践\多模态实践代码\image_run_local_pro.py

# streamlit_app.py

# ... (import语句保持不变) ...

@st.cache_resource
def load_models():
    print("正在加载所有模型...")
    try:
        # 定义所有模型的相对路径
        clip_path = "./model/clip/clip-vit-large-patch14"
        image_lib_path = "./model/image_lib/val2017"
        caption_model_path = "./model/vit-gpt2-image-captioning"
        translator_model_path = "./model/nllb-200-distilled-600M"

        # --- 核心修改：为ImageRetrievalLocal也传入翻译模型路径 ---
        image_retrieval = ImageRetrievalLocal(
            clip_model_path=clip_path,
            image_folder=image_lib_path,
            translator_model_path=translator_model_path  # 添加此行
        )

        image_captioning = ImageCaptioningLocal(
            caption_model_path=caption_model_path,
            translator_model_path=translator_model_path
        )
        print("所有模型加载成功。")
        return image_retrieval, image_captioning
    except Exception as e:
        st.error(f"模型加载失败，请检查模型路径是否正确。错误信息: {e}")
        sys.exit()


# ... (文件余下部分保持不变) ...


# 加载模型
image_retrieval_model, image_captioning_model = load_models()

# --- Streamlit 界面构建 (此部分无需任何修改) ---
st.set_page_config(page_title="多模态交互系统", layout="wide")

# vvvvv 在这里添加下面的代码 vvvvv
# 在右上角添加姓名和学号
col1, col2 = st.columns([3, 1])  # 创建两列，左边宽一些，右边窄一些

with col1:
    st.title("多模态交互系统")

with col2:
    # 使用Markdown和HTML/CSS来格式化和定位个人信息
    st.markdown(
        """
        <style>
        .info-container {
            /* 增加一点顶部的填充，使其与左侧标题的基线对齐 */
            padding-top: 2.75rem;
            text-align: right; /* 文本右对齐 */
        }
        .info-container p {
            color: #000000; /* 字体颜色设置为纯黑 */
            margin-bottom: 0; /* 移除段落的默认底部边距 */
            line-height: 1.5; /* 调整行高，使两行文字更紧凑 */
        }
        </style>
        <div class="info-container">
            <p>姓名：林鑫科<br>学号：1193220223</p>
        </div>
        """,
        unsafe_allow_html=True
    )
# ^^^^^ 在这里结束添加的代码 ^^^^^

st.markdown("一个基于本地化模型的图像描述生成与文本检索图像系统。")

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