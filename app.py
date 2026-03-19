import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from datetime import datetime

# 页面设置
st.set_page_config(page_title="听听你的画", page_icon="🎨", layout="wide")

# 标题
st.title("🎨 听听你的画")
st.markdown("画 ➡️ 音乐prompt生成器")
st.caption("上传你的画，听听它的声音")

# 初始化历史记录
if 'history' not in st.session_state:
    st.session_state.history = []

# 上传图片
uploaded_file = st.file_uploader("选择一张名画图片", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 显示原图
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ 上传的图片")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    # 转成OpenCV格式
    img_array = np.array(image)
    # 如果是RGBA格式转成RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    # 转成BGR（OpenCV用BGR）
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    # 开始分析按钮
    if st.button("🎵 开始分析并生成Prompt"):
        with st.spinner('正在分析画作特征...'):
            
            # ----- 1. 亮度分析 -----
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_std = np.std(gray)
            
            # ----- 2. 对比度分析 -----
            contrast = brightness_std
            
            # ----- 3. 色彩分析 -----
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            hue_mean = np.mean(hsv[:,:,0])
            saturation_mean = np.mean(hsv[:,:,1])
            value_mean = np.mean(hsv[:,:,2])
            
            # ----- 4. 简单笔触密度（用边缘检测模拟）-----
            edges = cv2.Canny(gray, 50, 150)
            brush_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            brush_density = min(brush_density, 1.0)  # 限制在0-1之间
            
            # ----- 5. 色彩复杂度（用颜色种类估算）-----
            pixels = img_bgr.reshape(-1, 3)
            # 简单去重（取前两位，近似颜色）
            unique_colors = len(set(tuple(p[:2]) for p in pixels[:1000:10]))
            color_complexity = min(unique_colors / 100, 1.0)
            
            # 收集所有特征
            features = {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'hue': float(hue_mean),
                'saturation': float(saturation_mean),
                'value': float(value_mean),
                'brush_density': float(brush_density),
                'color_complexity': float(color_complexity)
            }
            
            # 在右边显示特征
            with col2:
                st.subheader("📊 提取的视觉特征")
                
                # 显示所有参数的数值卡片
                col_metrics1, col_metrics2 = st.columns(2)
                with col_metrics1:
                    st.metric("亮度", f"{brightness:.1f}")
                    st.metric("对比度", f"{contrast:.1f}")
                    st.metric("色相", f"{hue_mean:.1f}")
                with col_metrics2:
                    st.metric("饱和度", f"{saturation_mean:.1f}")
                    st.metric("明度", f"{value_mean:.1f}")
                    st.metric("笔触密度", f"{brush_density:.3f}")
                
                # 显示进度条
                st.caption("📊 特征可视化")
                
                col_bar1, col_bar2 = st.columns(2)
                with col_bar1:
                    st.caption("亮度")
                    st.progress(min(brightness/255, 1.0))
                    st.caption("对比度")
                    st.progress(min(contrast/100, 1.0))
                    st.caption("饱和度") 
                    st.progress(min(saturation_mean/255, 1.0))
                with col_bar2:
                    st.caption("色相分布")
                    st.progress(min(hue_mean/180, 1.0))
                    st.caption("笔触密度")
                    st.progress(brush_density)  # 已经用min限制过了
                    st.caption("色彩复杂度")
                    st.progress(color_complexity)
                
                # 保留JSON数据（方便复制）
                with st.expander("查看原始数据"):
                    st.json(features)
            
            # ----- 6. 生成音乐Prompt -----
            
            # 根据特征映射音乐描述
            # 亮度决定情绪
            if brightness > 200:
                mood = "bright and joyful"
            elif brightness > 150:
                mood = "warm and peaceful"
            elif brightness > 100:
                mood = "mysterious and calm"
            else:
                mood = "dark and dramatic"
            
            # 对比度决定乐器
            if contrast > 70:
                instrument = "full orchestra with powerful dynamics"
            elif contrast > 50:
                instrument = "chamber ensemble with rich textures"
            else:
                instrument = "solo piano with subtle variations"
            
            # 色相决定调式
            if hue_mean > 150:
                key = "in minor key, melancholic"
            elif hue_mean > 80:
                key = "in modal style, meditative"
            else:
                key = "in major key, uplifting"
            
            # 饱和度决定和声密度
            if saturation_mean > 150:
                harmony = "dense harmonies and complex chords"
            elif saturation_mean > 80:
                harmony = "moderate harmonic movement"
            else:
                harmony = "simple, open harmonies"
            
            # 笔触密度决定节奏
            if brush_density > 0.3:
                rhythm = "fast, intricate rhythmic patterns"
            elif brush_density > 0.15:
                rhythm = "moderate, flowing rhythm"
            else:
                rhythm = "slow, sustained notes"
            
            # 色彩复杂度决定声部数量
            if color_complexity > 0.7:
                voices = "multiple interweaving melodic lines"
            elif color_complexity > 0.4:
                voices = "two or three distinct voices"
            else:
                voices = "single melodic line with accompaniment"
            
            # 组合成三个版本的prompt
            prompts = {
                "简洁版": f"{mood} {key} piece, {rhythm}",
                "详细版": f"A {mood} orchestral composition {key}, featuring {instrument}. The music has {harmony} with {rhythm} and {voices}.",
                "艺术版": f"Inspired by the visual qualities of the artwork: {mood} atmosphere with {harmony}. The {rhythm} evokes the painting's texture, while {voices} mirrors its color complexity."
            }
            
            # 显示prompt
            st.subheader("📝 生成的Suno Prompt")
            
            tab1, tab2, tab3 = st.tabs(["简洁版", "详细版", "艺术版"])
            
            with tab1:
                st.code(prompts["简洁版"], language="text")
                st.button("📋 复制简洁版", key="copy1")
            
            with tab2:
                st.code(prompts["详细版"], language="text")
                st.button("📋 复制详细版", key="copy2")
            
            with tab3:
                st.code(prompts["艺术版"], language="text")
                st.button("📋 复制艺术版", key="copy3")
            
            # 保存到历史
            st.session_state.history.append({
                'filename': uploaded_file.name,
                'prompts': prompts,
                'time': datetime.now().strftime("%H:%M:%S")
            })

# 显示历史记录
if st.session_state.history:
    with st.expander("📜 最近生成的记录"):
        for item in reversed(st.session_state.history[-5:]):
            st.write(f"**{item['filename']}** - {item['time']}")
            st.text(item['prompts']['简洁版'])
            st.divider()

# 页脚
st.markdown("---")
st.caption("✨ 上传你的画，听听它的声音")