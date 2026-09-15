import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import json
import os

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Waste Sorter Korea",
    page_icon="♻️",
    layout="centered"
)

# --- 2. LOCALIZATION & UI CONFIGURATION ---
LOCALES = {
    "English": {
        "title": "♻️ Waste Sorter in South Korea",
        "subtitle": "Upload a photo of an item to learn how to properly dispose of it according to Korean local district rules.",
        "select_lang": "Language / 언어",
        "select_district": "📍 Select your district in Korea:",
        "upload_label": "📸 Upload a photo of the item:",
        "uploaded_caption": "Uploaded Photo",
        "analyzing_spinner": "Analyzing image with AI model...",
        "btn_analyze": "Classify Waste 🔍",
        "analysis_complete": "Analysis Complete!",
        "result_cat": "Category:",
        "result_conf": "Model Confidence:",
        "result_rule": "Local Disposal Instructions:",
        "no_rule_found": "Instructions for this district are currently unavailable.",
        "bag_info_header": "🎒 District Disposal Details:",
        "general_bag": "General Waste Bag:",
        "food_bag": "Food Waste Bag:",
        "pickup_days": "Pickup Days:",
        "disposal_time": "Disposal Hours:",
        "tips_header": "💡 Tips for better classification results:",
        "tips": [
            "Take a clear photo with good lighting.",
            "Center a single object in the frame.",
            "Avoid dark or heavily cluttered backgrounds."
        ],
        "categories": {
            "plastic": "Plastic 🥤",
            "paper": "Paper / Cardboard 📦",
            "glass": "Glass 🍾",
            "metal": "Metal 🥫",
            "organic": "Organic / Food Waste 🍎",
            "general": "General Waste 🗑️"
        }
    },
    "한국어": {
        "title": "♻️ 한국 분리수거 가이드",
        "subtitle": "쓰레기 사진을 업로드하여 해당 지역의 올바른 분리배출 방법을 확인하세요.",
        "select_lang": "Language / 언어",
        "select_district": "📍 거주하는 지역을 선택하세요:",
        "upload_label": "📸 쓰레기 사진을 업로드하세요:",
        "uploaded_caption": "업로드된 사진",
        "analyzing_spinner": "인공지능 모델이 이미지를 분석 중입니다...",
        "btn_analyze": "분류하기 🔍",
        "analysis_complete": "분석 완료!",
        "result_cat": "분류 카테고리:",
        "result_conf": "모델 신뢰도:",
        "result_rule": "지역별 배출 방법 안내:",
        "no_rule_found": "해당 지역의 배출 지침 정보가 없습니다.",
        "bag_info_header": "🎒 지역별 종량제 봉투 및 배출 정보:",
        "general_bag": "일반 쓰레기 봉투:",
        "food_bag": "음식물 쓰레기 봉투:",
        "pickup_days": "배출 요일:",
        "disposal_time": "배출 시간:",
        "tips_header": "💡 정확한 인식 결과를 위한 팁:",
        "tips": [
            "밝은 조명에서 명확하게 촬영해 주세요.",
            "한 번에 하나의 물체만 중앙에 위치시켜 주세요.",
            "어둡거나 복잡한 배경을 피해 주세요."
        ],
        "categories": {
            "plastic": "플라스틱 🥤",
            "paper": "종이류 / 상자 📦",
            "glass": "유리병 🍾",
            "metal": "캔 / 금속류 🥫",
            "organic": "음식물 쓰레기 🍎",
            "general": "일반 쓰레기 🗑️"
        }
    }
}

# Color mapping for category UI cards
CATEGORY_COLORS = {
    "plastic": {"bg": "#e6f4ea", "border": "#34a853", "label": "🟢 Plastic Recyclable"},
    "paper": {"bg": "#e8f0fe", "border": "#4285f4", "label": "🔵 Paper Recyclable"},
    "glass": {"bg": "#feefc3", "border": "#fbbc04", "label": "🟡 Glass Recyclable"},
    "metal": {"bg": "#f1f3f4", "border": "#5f6368", "label": "⚪ Metal Recyclable"},
    "organic": {"bg": "#fce8e6", "border": "#ea4335", "label": "🔴 Food Waste"},
    "general": {"bg": "#f1f3f4", "border": "#9aa0a6", "label": "⚪ General Waste (Non-recyclable)"}
}

# --- 3. LOAD DISTRICT RULES ---
@st.cache_data
def load_district_rules():
    path = os.path.join("data", "district_rules.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# --- 4. LOAD MODEL AND PREDICTION LOGIC ---
@st.cache_resource
def load_model():
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()
    return model, weights

model, weights = load_model()
preprocess = weights.transforms()

IMAGENET_TO_WASTE = {
    "water_bottle": "plastic", "pop_bottle": "plastic", "plastic_bag": "plastic",
    "water_jug": "plastic", "pill_bottle": "plastic", "hair_spray": "plastic",
    "carton": "paper", "envelope": "paper", "paper_towel": "paper",
    "notebook": "paper", "book": "paper", "binder": "paper", "comic_book": "paper",
    "menu": "paper", "book_jacket": "paper", "web_site": "paper", "crossword": "paper",
    "wine_bottle": "glass", "beer_bottle": "glass", "glass": "glass", "goblet": "glass",
    "can": "metal", "tin_can": "metal", "soup_bowl": "metal", "brass": "metal",
    "banana": "organic", "apple": "organic", "orange": "organic", "lemon": "organic",
    "fig": "organic", "pineapple": "organic", "pomegranate": "organic", "strawberry": "organic"
}

def predict_waste_type(image):
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    top_prob, top_cat_id = torch.topk(probabilities, 5)
    
    waste_category = "general"
    confidence = float(top_prob[0]) * 100
    
    for i in range(5):
        cat_name = weights.meta["categories"][top_cat_id[i].item()].lower()
        for key, val in IMAGENET_TO_WASTE.items():
            if key in cat_name:
                waste_category = val
                confidence = float(top_prob[i]) * 100
                return waste_category, confidence
                
    return waste_category, float(top_prob[0]) * 100

# --- 5. APPLICATION INTERFACE ---

st.sidebar.title("Settings / 설정")
lang_choice = st.sidebar.selectbox("Language / 언어", list(LOCALES.keys()))
t = LOCALES[lang_choice]

st.title(t["title"])
st.write(t["subtitle"])
st.markdown("---")

district_data = load_district_rules()
districts = list(district_data.keys()) if district_data else ["1. Gangnam-gu", "2. Mapo-gu"]
selected_district = st.selectbox(t["select_district"], districts)

# Показываем справочные данные района из файлов/скриншотов
if selected_district in district_data:
    info = district_data[selected_district].get("district_info", {})
    if info:
        st.markdown(
            f"""
            > *{t['bag_info_header']}*  
            > ⚪ *{t['general_bag']}* {info.get('general_bag', 'Белый')}  
            > 🟡 *{t['food_bag']}* {info.get('food_bag', 'Стандартный')}  
            > 📅 *{t['pickup_days']}* {info.get('pickup_days', 'Вс — Чт')}  
            > ⏰ *{t['disposal_time']}* {info.get('disposal_time', '19:00 — 23:00')}
            """
        )

uploaded_file = st.file_uploader(t["upload_label"], type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info(f"*{t['tips_header']}*\n" + "\n".join([f"- {tip}" for tip in t["tips"]]))

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=t["uploaded_caption"], use_container_width=True)
    
    if st.button(t["btn_analyze"], type="primary", use_container_width=True):
        with st.spinner(t["analyzing_spinner"]):
            category_key, confidence = predict_waste_type(image)
            
            cat_display = t["categories"].get(category_key, category_key)
            rule_text = district_data.get(selected_district, {}).get(
                category_key, t["no_rule_found"]
            )
            color_theme = CATEGORY_COLORS.get(category_key, CATEGORY_COLORS["general"])
            
            st.success(t["analysis_complete"])
            
            st.markdown(
                f"""
                <div style="background-color:{color_theme['bg']}; border-left: 6px solid {color_theme['border']}; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <h3 style="margin: 0; color: #1f2937;">{t['result_cat']} {cat_display}</h3>
                    <p style="margin: 5px 0 0 0; font-weight: bold; color: {color_theme['border']};">{color_theme['label']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown(f"*{t['result_conf']} {confidence:.1f}%*")
            st.progress(min(int(confidence), 100))
            
            st.info(f"*{t['result_rule']}*\n\n{rule_text}")