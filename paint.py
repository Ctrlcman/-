import torch
import streamlit as st
from diffusers import StableDiffusionPipeline
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image

# ------------------ 模型加载 ------------------
@st.cache_resource
def load_sd_pipeline():
    model_id = "stabilityai/stable-diffusion-2-1-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    return pipe, device

@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-zh-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

pipe, device = load_sd_pipeline()
tokenizer, translator_model = load_translator()

# ------------------ 中文转英文函数 ------------------
def translate_zh2en(text):
    batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    gen = translator_model.generate(**batch, max_length=100)
    translated = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return translated

# ------------------ 页面设置 ------------------
st.set_page_config(page_title="🎨灵魂画手", layout="centered")
st.title("🧠灵魂画手")

st.markdown("输入中文提示词，灵魂画手开始作画 📷")

# ------------------ 用户输入 ------------------
prompt_zh = st.text_input("📝 你希望画什么？", value="一只猫在看星空")
negative_zh = st.text_input("🚫 不希望出现的内容", value="模糊、低质量")
steps = st.slider("🎨 画图精致度", 10, 50, 30)
guidance = st.slider("📢 描述影响强度", 1.0, 15.0, 7.5)
width = st.slider("🖼️ 图片宽度（像素）", 256, 768, 512, step=64)
height = st.slider("🖼️ 图片高度（像素）", 256, 768, 512, step=64)

# ------------------ 生成按钮 ------------------
if st.button("🎨 生成图片"):
    with st.spinner("⏳灵魂画手正在绘制，请稍候..."):
        prompt_en = translate_zh2en(prompt_zh)
        negative_en = translate_zh2en(negative_zh)
        generator = torch.Generator(device=device).manual_seed(42)
        output = pipe(
            prompt=prompt_en,
            negative_prompt=negative_en,
            width=width,
            height=height,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator
        )
        image = output.images[0]
        st.image(image, use_column_width=True)
