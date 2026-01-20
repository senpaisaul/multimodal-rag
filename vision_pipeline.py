import base64
import io
import json
from PIL import Image

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from schemas import VisionFact


VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
MAX_PIXELS = 30_000_000
MIN_PIXELS = 20_000  # filter icons/logos


def resize_image_if_needed(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    pixels = w * h

    if pixels < MIN_PIXELS:
        raise ValueError("Image too small to analyze")

    if pixels <= MAX_PIXELS:
        return image_bytes

    scale = (MAX_PIXELS / pixels) ** 0.5
    new_size = (int(w * scale), int(h * scale))
    img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


class VisionPipeline:
    def __init__(self, groq_key: str):
        self.llm = ChatGroq(
            api_key=groq_key,
            model=VISION_MODEL,
            temperature=0
        )

    def analyze_image(self, image_bytes: bytes, page: int) -> VisionFact:
        try:
            safe_bytes = resize_image_if_needed(image_bytes)
        except ValueError:
            return None  # skip useless images

        encoded = base64.b64encode(safe_bytes).decode("utf-8")

        prompt = HumanMessage(content=[
            {
                "type": "text",
                "text": """
You are an OCR + vision extraction system.

Return STRICT JSON ONLY.
No markdown. No commentary.

Schema:
{
  "image_type": "chart|table|diagram|text|other",
  "description": "",
  "x_label": "",
  "y_label": "",
  "data_points": [[x,y]],
  "trend": "",
  "confidence": "high|medium|low"
}
"""
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"}
            }
        ])

        raw = self.llm.invoke([prompt]).content

        try:
            data = json.loads(raw)
            fact = VisionFact(**data, page=page)
            return fact
        except Exception:
            return VisionFact(
                page=page,
                image_type="other",
                description="Image detected but could not be parsed reliably.",
                confidence="low"
            )

