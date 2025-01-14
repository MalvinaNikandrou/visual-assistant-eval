from enum import Enum

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
    "ar": "لا يمكن الرد عليها" , #Arabic
    "bn": "উত্তর দেওয়া সম্ভব নয়", #Bengali
    "cs": "nelze odpovědět", #Czech
    "da": "kan ikke besvares", #Danish
    "de": "unbeantwortbar", #German
    "el": "δεν μπορεί να απαντηθεί", #Greek
    "en": "unanswerable", #English
    "es": "no se puede responder", #Spanish
    "fa": "قابل به پاسخ نیست", #Persian
    "fi": "ei voida vastata", #Finnish
    "fil": "hindi masasagot ", #Filipino"  # Tagalog
    "fr": "impossible à répondre", #French
    "he": "לא ניתן לענות", #Hebrew"  # , appears as iw in mC4 languages and in Wikipedia
    "hi": "उत्तर नहीं दिया जा सकता", #Hindi
    "hr": "ne može se odgovoriti", #Croatian" or "Neodgovoriv
    "hu": "megválaszolhatatlan", #Hungarian
    "id": "tidak dapat dijawab", #Indonesian / Malay
    "it": "non si può rispondere", #Italian
    "ja": "答えられない", #Japanese
    "ko": "대답할 수 없는", #Korean
    "mi": "kāore e taea te whakautu", #Maori
    "nl": "onbeantwoordbaar", #Dutch
    "no": "kan ikke besvares.", #Norwegian"  # Google translate has only a Norwegian option, we include Bokmål
    "pl": "nie można odpowiedzieć", #Polish
    "pt": "irrespondível", #Portuguese
    "quz": "mana atiyman kutichiy", #Cusco Quechua  # not in mt5 -latin script
    "ro": "fără răspuns", #Romanian
    "ru": "неотвечаемый", #Russian
    "sv": "omöjlig att besvara", #Swedish or "Kan inte besvaras"
    "sw": "isiyoweza kujibiwa", #Swahili"  # Arabic script, similar punctuation to english
    "te": "సమాధానం చెప్పలేని", #Telugu
    "th": "ตอบไม่ได้", #Thai
    "tr": "cevaplanamaz", #Turkish
    "uk": "hе можна відповісти", #Ukrainian
    "vi": "không thể trả lời được", #Vietnamese
    "zh": "无法回答", #Chinese
    
}


lang_flores200_codes = {
    Language.ar: "arb_Arab",
    Language.bn: "ben_Beng",
    Language.cs: "ces_Latn",
    Language.da: "dan_Latn",
    Language.de: "deu_Latn",
    Language.el: "ell_Grek",
    Language.en: "eng_Latn",
    Language.es: "spa_Latn",
    Language.fa: "pes_Arab",
    Language.fi: "fin_Latn",
    Language.fil: "tgl_Latn",
    Language.fr: "fra_Latn",
    Language.he: "heb_Hebr",
    Language.hi: "hin_Deva",
    Language.hr: "hrv_Latn",
    Language.hu: "hun_Latn",
    Language.id: "ind_Latn",
    Language.it: "ita_Latn",
    Language.ja: "jpn_Jpan",
    Language.ko: "kor_Hang",
    Language.mi: "mri_Latn",
    Language.nl: "nld_Latn",
    Language.no: "nob_Latn",
    Language.pl: "pol_Latn",
    Language.pt: "por_Latn",
    Language.ro: "ron_Latn",
    Language.ru: "rus_Cyrl",
    Language.sv: "swe_Latn",
    Language.sw: "swh_Latn",
    Language.te: "tel_Telu",
    Language.th: "tha_Thai",
    Language.tr: "tur_Latn",
    Language.uk: "ukr_Cyrl",
    Language.vi: "vie_Latn",
    Language.zh: "zho_Hans",  # Chinese simplified
}

