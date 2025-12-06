from enum import Enum
from pathlib import Path
from typing import Type

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from ..utils.json_parsing import parse_pydantic_schema
from .dataset_base import VQADataset

SEED = 20039


class Language(Enum):
    ar = "Arabic"
    bn = "Bengali"
    cs = "Czech"
    da = "Danish"
    de = "German"
    el = "Greek"
    en = "English"
    es = "Spanish"
    fa = "Persian"
    fi = "Finnish"
    fil = "Filipino"  # Tagalog
    fr = "French"
    he = "Hebrew"  # , appears as iw in mC4 languages and in Wikipedia
    hi = "Hindi"
    hr = "Croatian"  # , not in mt5 but written with Latin characters / related to serbian which is in mt5
    hu = "Hungarian"
    id = "Indonesian"
    it = "Italian"
    ja = "Japanese"
    ko = "Korean"
    mi = "Maori"
    nl = "Dutch"
    no = "Norwegian"  # Google translate has only a Norwegian option, we include Bokmål
    pl = "Polish"
    pt = "Portuguese"
    quz = "Cusco Quechua"  # not in mt5 -latin script"
    ro = "Romanian"
    ru = "Russian"
    sv = "Swedish"
    sw = "Swahili"  # Arabic script, similar punctuation to english"
    te = "Telugu"
    th = "Thai"
    tr = "Turkish"
    uk = "Ukrainian"
    vi = "Vietnamese"
    zh = "Chinese"

    @classmethod
    def from_code(cls, code: str) -> "Language":
        return cls[code]

    @classmethod
    def get_language_from_code(cls, code):
        for item in cls:
            if code == item.name:
                return item.value
        return None

    @classmethod
    def language_uses_whitespaces(cls, code):
        return code not in {"ja", "zh", "th"}  # Japanese  # Mandarin  # Thai

    @classmethod
    def languages_after_translation(cls):
        return [item for item in cls if item.name != "quz"]


# Translation of "unasnwerable"
UnanswerableMapping = {
    "ar": "لا يمكن الرد عليها",  # Arabic
    "bn": "উত্তর দেওয়া সম্ভব নয়",  # Bengali
    "cs": "nelze odpovědět",  # Czech
    "da": "kan ikke besvares",  # Danish
    "de": "unbeantwortbar",  # German
    "el": "δεν μπορεί να απαντηθεί",  # Greek
    "en": "unanswerable",  # English
    "es": "no se puede responder",  # Spanish
    "fa": "قابل به پاسخ نیست",  # Persian
    "fi": "ei voida vastata",  # Finnish
    "fil": "hindi masasagot ",  # Filipino"  # Tagalog
    "fr": "impossible à répondre",  # French
    "he": "לא ניתן לענות",  # Hebrew"  # , appears as iw in mC4 languages and in Wikipedia
    "hi": "उत्तर नहीं दिया जा सकता",  # Hindi
    "hr": "ne može se odgovoriti",  # Croatian" or "Neodgovoriv
    "hu": "megválaszolhatatlan",  # Hungarian
    "id": "tidak dapat dijawab",  # Indonesian / Malay
    "it": "non si può rispondere",  # Italian
    "ja": "答えられない",  # Japanese
    "ko": "대답할 수 없는",  # Korean
    "mi": "kāore e taea te whakautu",  # Maori
    "nl": "onbeantwoordbaar",  # Dutch
    "no": "kan ikke besvares.",  # Norwegian"  # Google translate has only a Norwegian option, we include Bokmål
    "pl": "nie można odpowiedzieć",  # Polish
    "pt": "irrespondível",  # Portuguese
    "quz": "mana atiyman kutichiy",  # Cusco Quechua  # not in mt5 -latin script
    "ro": "fără răspuns",  # Romanian
    "ru": "неотвечаемый",  # Russian
    "sv": "omöjlig att besvara",  # Swedish or "Kan inte besvaras"
    "sw": "isiyoweza kujibiwa",  # Swahili"  # Arabic script, similar punctuation to english
    "te": "సమాధానం చెప్పలేని",  # Telugu
    "th": "ตอบไม่ได้",  # Thai
    "tr": "cevaplanamaz",  # Turkish
    "uk": "hе можна відповісти",  # Ukrainian
    "vi": "không thể trả lời được",  # Vietnamese
    "zh": "无法回答",  # Chinese
}


class VQAResponse(PydanticBaseModel):
    answer: str = Field(description="Answer for the visual question")


class VizWizVQADataset(VQADataset):
    name = "vizwiz_vqa"
    json_schema: Type[PydanticBaseModel] = VQAResponse


class MultilingualVizWizVQADataset(VQADataset):
    name = "multilingual_image_question_answering"
    json_schema: Type[PydanticBaseModel] = VQAResponse

    def __init__(self, images_path: str, path: str, template_name: str):
        super().__init__(images_path, path, template_name)
        lang = Path(path).stem.split("_")[-1]
        self.lang = Language.from_code(lang)
        self.use_language_code = "paligemma" in template_name

    def get_prompt(self, question: str) -> str:
        if self.use_language_code:
            lang = self.lang.name
        else:
            lang = self.lang.value
        return self.template.render(
            question=question,
            lang=lang,
            unanswerable=UnanswerableMapping.get(self.lang.name),
            json_schema=parse_pydantic_schema(self.json_schema),
        )

    def get_answers(self, answers: list[str]) -> list[str]:
        if "translated_answer" in answers[0]:
            return [answer["translated_answer"] for answer in answers]

        ground_truth = [answer["answer"] for answer in answers]
        return ground_truth
