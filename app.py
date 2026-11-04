import os
import base64
from fastapi import FastAPI, UploadFile, File, Form
from huggingface_hub import InferenceClient

app = FastAPI(title="My Smart AI Server on Render")

# يأخذ المفتاح المجاني بأمان من إعدادات Render
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

# الموديل الشامل العملاق للصور والنصوص والبيانات الزراعية
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"

@app.get("/")
def home():
    return {"status": "online", "message": "السيرفر يعمل بنجاح! اذهب إلى /docs للتجربة"}

@app.post("/analyze")
async def analyze(
    prompt: str = Form(..., description="سؤالك أو البيانات الزراعية"),
    image: UploadFile = File(None, description="صورة النبتة (اختياري)")
):
    messages = [
        {"role": "system", "content": "أنت خبير ذكاء اصطناعي شامل ومختص بتحليل الصور والبيانات الزراعية والعامة بدقة باللغة العربية."}
    ]

    if image is not None:
        img_bytes = await image.read()
        base64_img = base64.b64encode(img_bytes).decode("utf-8")
        img_url = f"data:image/jpeg;base64,{base64_img}"
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": prompt}
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=600,
        temperature=0.3
    )

    return {
        "status": "success",
        "reply": response.choices[0].message.content
    }
