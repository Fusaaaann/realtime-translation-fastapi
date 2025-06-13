import json
import os
from enum import Enum
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Language(Enum):
    ENGLISH = "ENGLISH"
    CHINESE = "CHINESE"
    THAI = "THAI"
    VIETNAMESE = "VIETNAMESE"
    INDONESIAN = "INDONESIAN"
    MALAY = "MALAY"
    TAGALOG = "TAGALOG"
    BURMESE = "BURMESE"
    KHMER = "KHMER"
    LAO = "LAO"
    HINDI = "HINDI"
    TAMIL = "TAMIL"
    BENGALI = "BENGALI"
    TELUGU = "TELUGU"
    MALAYALAM = "MALAYALAM"
    SINHALESE = "SINHALESE"

    @property
    def native_name(self) -> str:
        mapping = {
            Language.ENGLISH: "English",
            Language.CHINESE: "中文",
            Language.VIETNAMESE: "Tiếng Việt",
            Language.THAI: "ไทย",
            Language.INDONESIAN: "Bahasa Indonesia",
            Language.MALAY: "Bahasa Melayu",
            Language.TAGALOG: "Tagalog",
            Language.BURMESE: "မြန်မာဘာသာ",
            Language.KHMER: "ភាសាខ្មែរ",
            Language.LAO: "ພາສາລາວ",
            Language.HINDI: "हिन्दी",
            Language.TAMIL: "தமிழ்",
            Language.BENGALI: "বাংলা",
            Language.TELUGU: "తెలుగు",
            Language.MALAYALAM: "മലയാളം",
            Language.SINHALESE: "සිංහල",
        }
        return mapping.get(self, self.value.title())


# Dictionary with localized messages for each language
LOCALIZED_MESSAGES = {
    "ENGLISH": {
        "waiting": "Waiting for captions to begin...",
        "switching": "Switching language... waiting for new captions",
        "instructions": "Click anywhere to enable audio playback",
    },
    "CHINESE": {"waiting": "等待字幕开始...", "switching": "正在切换语言...等待新字幕", "instructions": "点击任意位置启用音频播放"},
    "THAI": {"waiting": "กำลังรอคำบรรยายเริ่มต้น...", "switching": "กำลังเปลี่ยนภาษา...รอคำบรรยายใหม่", "instructions": "คลิกที่ใดก็ได้เพื่อเปิดใช้งานการเล่นเสียง"},
    "VIETNAMESE": {
        "waiting": "Đang chờ phụ đề bắt đầu...",
        "switching": "Đang chuyển ngôn ngữ... chờ phụ đề mới",
        "instructions": "Nhấp vào bất kỳ đâu để bật phát âm thanh",
    },
    "INDONESIAN": {
        "waiting": "Menunggu teks mulai...",
        "switching": "Mengubah bahasa... menunggu teks baru",
        "instructions": "Klik di mana saja untuk mengaktifkan pemutaran audio",
    },
    "MALAY": {
        "waiting": "Menunggu sari kata bermula...",
        "switching": "Menukar bahasa... menunggu sari kata baharu",
        "instructions": "Klik di mana-mana untuk membolehkan main balik audio",
    },
    "TAGALOG": {
        "waiting": "Naghihintay para magsimula ang mga caption...",
        "switching": "Nagpapalit ng wika... naghihintay ng bagong caption",
        "instructions": "Mag-click kahit saan para paganahin ang audio playback",
    },
    "BURMESE": {
        "waiting": "စာတန်းထိုးများ စတင်ရန် စောင့်ဆိုင်းနေသည်...",
        "switching": "ဘာသာစကား ပြောင်းနေသည်... စာတန်းထိုးအသစ်များကို စောင့်ဆိုင်းနေသည်",
        "instructions": "အသံဖွင့်ရန် မည်သည့်နေရာတွင်မဆို နှိပ်ပါ",
    },
    "KHMER": {"waiting": "កំពុងរង់ចាំចំណងជើងរងចាប់ផ្តើម...", "switching": "កំពុងប្តូរភាសា... កំពុងរង់ចាំចំណងជើងរងថ្មី", "instructions": "ចុចកន្លែងណាមួយដើម្បីបើកការចាក់សំឡេង"},
    "LAO": {"waiting": "ກຳລັງລໍຖ້າຄຳບັນຍາຍເລີ່ມຕົ້ນ...", "switching": "ກຳລັງປ່ຽນພາສາ... ລໍຖ້າຄຳບັນຍາຍໃໝ່", "instructions": "ຄລິກບ່ອນໃດກໍໄດ້ເພື່ອເປີດການຫຼິ້ນສຽງ"},
    "HINDI": {
        "waiting": "कैप्शन शुरू होने का इंतज़ार कर रहे हैं...",
        "switching": "भाषा बदल रहे हैं... नए कैप्शन का इंतज़ार कर रहे हैं",
        "instructions": "ऑडियो प्लेबैक सक्षम करने के लिए कहीं भी क्लिक करें",
    },
    "TAMIL": {
        "waiting": "தலைப்புகள் தொடங்க காத்திருக்கிறது...",
        "switching": "மொழியை மாற்றுகிறது... புதிய தலைப்புகளுக்காக காத்திருக்கிறது",
        "instructions": "ஒலி இயக்கத்தை இயக்க எங்கேயும் கிளிக் செய்யவும்",
    },
    "BENGALI": {
        "waiting": "ক্যাপশন শুরু হওয়ার জন্য অপেক্ষা করছে...",
        "switching": "ভাষা পরিবর্তন করা হচ্ছে... নতুন ক্যাপশনের জন্য অপেক্ষা করছে",
        "instructions": "অডিও চালানো সক্ষম করতে যেকোনো জায়গায় ক্লিক করুন",
    },
    "TELUGU": {
        "waiting": "శీర్షికలు ప్రారంభం కావడానికి వేచి ఉంది...",
        "switching": "భాష మారుస్తోంది... కొత్త శీర్షికల కోసం వేచి ఉంది",
        "instructions": "ఆడియో ప్లేబ్యాక్‌ని ప్రారంభించడానికి ఎక్కడైనా క్లిక్ చేయండి",
    },
    "MALAYALAM": {
        "waiting": "അടിക്കുറിപ്പുകൾ ആരംഭിക്കാൻ കാത്തിരിക്കുന്നു...",
        "switching": "ഭാഷ മാറ്റുന്നു... പുതിയ അടിക്കുറിപ്പുകൾക്കായി കാത്തിരിക്കുന്നു",
        "instructions": "ഓഡിയോ പ്ലേബാക്ക് പ്രവർത്തനക്ഷമമാക്കാൻ എവിടെയെങ്കിലും ക്ലിക്കുചെയ്യുക",
    },
    "SINHALESE": {
        "waiting": "ශීර්ෂ පාඨ ආරම්භ වීමට රැඳී සිටිමින්...",
        "switching": "භාෂාව මාරු කරමින්... නව ශීර්ෂ පාඨ සඳහා රැඳී සිටිමින්",
        "instructions": "ශ්‍රව්‍ය ධාවනය සක්‍රීය කිරීමට ඕනෑම තැනක ක්ලික් කරන්න",
    },
}

# Default messages for languages without translations
DEFAULT_MESSAGES = {
    "waiting": "Waiting for captions to begin...",
    "switching": "Switching language... waiting for new captions",
    "instructions": "Click anywhere to enable audio playback",
}


translation_prompt = {}
special_terms = {}  # Single global dict for special terms

# Configuration files
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompt.json")
SPECIAL_TERMS_FILE = os.path.join(os.path.dirname(__file__), "special_terms.json")


def convert_key(key_str):
    """Convert "ENGLISH-CHINESE" into (Language.ENGLISH, Language.CHINESE)"""
    source, target = key_str.split("-")
    return (getattr(Language, source.strip().upper()), getattr(Language, target.strip().upper()))


def convert_lang_pair_key(key_str):
    """Convert "en-zh" into normalized format"""
    source, target = key_str.upper().split("-")
    return f"{source}-{target}"


def load_prompt(file_path=PROMPT_FILE):
    """Load translation prompts from JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Prompt file {file_path} not found. Using defaults.")
        return
    except Exception as e:
        print(f"Error loading prompt from {file_path}: {e}")
        return

    translation_prompt.clear()
    for key_str, prompt in data.items():
        try:
            key = convert_key(key_str)
            translation_prompt[key] = prompt
        except (AttributeError, ValueError) as e:
            print(f"Invalid language key '{key_str}': {e}")
            continue

    print("Prompts loaded successfully.")


def load_special_terms(file_path=SPECIAL_TERMS_FILE):
    """Load special terms from JSON file - simplified to single domain"""
    global special_terms

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Special terms file {file_path} not found. Using empty terms.")
        special_terms = {}
        return
    except Exception as e:
        print(f"Error loading special terms from {file_path}: {e}")
        return

    # Simplified: directly load the single domain structure
    special_terms = data
    print("Special terms loaded successfully.")


def get_assembled_prompt(
    source_lang: str, target_lang: str, prompt_type: str, source_text: str = "", surface_translation: str = "", special_terms_context: str = ""
) -> str:
    """
    Assemble prompts for different translation passes.

    Args:
        source_lang: Source language code
        target_lang: Target language code
        prompt_type: 'single_pass', 'surface_translation', or 'refinement'
        source_text: Original text to translate
        surface_translation: Literal translation (for refinement pass)
        special_terms_context: Special terms context string

    Returns:
        Assembled prompt string
    """
    # Get prompt template
    key = (Language(source_lang.upper()), Language(target_lang.upper()))
    prompt_data = translation_prompt.get(key, {})

    # Default prompts if not found
    default_prompts = {
        "preprocessing": "",
        "surface_translation": f"Provide literal translation from {source_lang} to {target_lang}: ${{text}}",
        "refinement": f"Refine into natural {target_lang}: ${{surface_translation}}",
        "single_pass": f"Translate from {source_lang} to {target_lang}: ${{text}}",
    }

    # Get the specific subprompt
    subprompt = prompt_data.get(prompt_type, default_prompts[prompt_type])
    preprocessing = prompt_data.get("preprocessing", "")

    # Assemble the full prompt based on type
    if prompt_type == "single_pass":
        # Single-pass: preprocessing + single_pass + special terms
        full_prompt = ""
        if preprocessing:
            full_prompt += preprocessing + "\n\n"
        full_prompt += subprompt.replace("${text}", source_text)
        if special_terms_context:
            full_prompt += special_terms_context

    elif prompt_type == "surface_translation":
        # Surface translation: preprocessing + surface_translation + special terms
        full_prompt = f"Translate {source_lang} to {target_lang} - LITERAL PASS:\n"
        if preprocessing:
            full_prompt += preprocessing + "\n"
        full_prompt += subprompt
        full_prompt += '\n\nExample: "I am going" → "Yo soy yendo" (literal, not "Yo voy")'
        full_prompt += f"\n\nText: {source_text}"
        if special_terms_context:
            full_prompt += special_terms_context
        full_prompt += "\n\nLiteral translation:"

    elif prompt_type == "refinement":
        # Refinement: refinement subprompt + context
        full_prompt = f"REFINEMENT PASS - Make natural {target_lang}:\n\n"
        full_prompt += f"Source: {source_text}\n"
        full_prompt += f"Literal: {surface_translation}\n\n"
        full_prompt += subprompt
        if special_terms_context:
            full_prompt += special_terms_context
        full_prompt += "\n\nNatural translation:"

    return full_prompt


def get_terms_for_lang_pair(source_lang: str, target_lang: str) -> dict:
    """
    Get special terms for a specific language pair from global special_terms.
    If source-target is not found, try target-source and reverse the key-value pairs.

    Args:
        source_lang: Source language code (e.g., 'en')
        target_lang: Target language code (e.g., 'zh')

    Returns:
        Dictionary of special terms for the language pair
    """
    source_lang = source_lang.upper()
    target_lang = target_lang.upper()

    # Try direct lookup first
    lang_pair_key = f"{source_lang}-{target_lang}"
    if lang_pair_key in special_terms:
        return special_terms[lang_pair_key]

    # Try reverse lookup if direct not found
    reverse_lang_pair_key = f"{target_lang}-{source_lang}"
    if reverse_lang_pair_key in special_terms:
        # Reverse the key-value pairs
        reversed_terms = {value: key for key, value in special_terms[reverse_lang_pair_key].items()}
        return reversed_terms

    # Return empty dict if neither direction is found
    return {}


class ConfigFileEventHandler(FileSystemEventHandler):
    def __init__(self, prompt_file=PROMPT_FILE, terms_file=SPECIAL_TERMS_FILE):
        self._prompt_file = prompt_file
        self._terms_file = terms_file

    def on_modified(self, event):
        if event.src_path.endswith(os.path.basename(self._prompt_file)):
            print(f"{self._prompt_file} modified. Reloading prompts...")
            load_prompt(self._prompt_file)
        elif event.src_path.endswith(os.path.basename(self._terms_file)):
            print(f"{self._terms_file} modified. Reloading special terms...")
            load_special_terms(self._terms_file)


def start_config_watcher(prompt_file=PROMPT_FILE, terms_file=SPECIAL_TERMS_FILE):
    """Start watching both config files for changes"""
    event_handler = ConfigFileEventHandler(prompt_file, terms_file)
    observer = Observer()

    # Watch directories containing the files
    prompt_dir = os.path.dirname(os.path.abspath(prompt_file))
    terms_dir = os.path.dirname(os.path.abspath(terms_file))

    observer.schedule(event_handler, prompt_dir, recursive=False)
    if terms_dir != prompt_dir:
        observer.schedule(event_handler, terms_dir, recursive=False)

    observer.start()
    print("Started watching config files for changes.")

    thread = threading.Thread(target=observer.join, daemon=True)
    thread.start()
    return observer


# Simplified utility functions
def create_special_terms_dict(*term_pairs, validate: bool = True) -> dict:
    """Helper function to create special terms dictionary with validation"""
    terms_dict = dict(term_pairs)

    if validate:
        for source, target in terms_dict.items():
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError(f"Terms must be strings: {source} → {target}")
            if not source.strip() or not target.strip():
                raise ValueError(f"Terms cannot be empty: '{source}' → '{target}'")

    return terms_dict


load_prompt(PROMPT_FILE)
load_special_terms(SPECIAL_TERMS_FILE)
start_config_watcher(PROMPT_FILE, SPECIAL_TERMS_FILE)
