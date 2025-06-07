from enum import Enum


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
translation_prompt = {
    (Language.ENGLISH, Language.CHINESE): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from English to Chinese. "
        "Provide only the final translated text:\n\n${text}"
    ),
    (Language.ENGLISH, Language.VIETNAMESE): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from English to Vietnamese. "
        "Provide only the final translated text:\n\n${text}"
    ),
    (Language.ENGLISH, Language.THAI): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from English to Thai. "
        "Provide only the final translated text:\n\n${text}"
    ),
    (Language.CHINESE, Language.ENGLISH): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from Chinese to English. "
        "Provide only the final translated text:\n\n${text}"
    ),
    (Language.CHINESE, Language.VIETNAMESE): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from Chinese to Vietnamese. "
        "Provide only the final translated text:\n\n${text}"
    ),
    (Language.CHINESE, Language.THAI): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from Chinese to Thai. "
        "Provide only the final translated text:\n\n${text}"
    ),
    (Language.VIETNAMESE, Language.ENGLISH): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from Vietnamese to English. "
        "Provide only the final translated text:\n\n${text}"
    ),
    (Language.VIETNAMESE, Language.CHINESE): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from Vietnamese to Chinese. "
        "Provide only the final translated text:\n\n${text}"
    ),
    (Language.VIETNAMESE, Language.THAI): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from Vietnamese to Thai. "
        "Provide only the final translated text:\n\n${text}"
    ),
    (Language.THAI, Language.ENGLISH): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from Thai to English. "
        "Provide only the final translated text:\n\n${text}"
    ),
    (Language.THAI, Language.CHINESE): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from Thai to Chinese. "
        "Provide only the final translated text:\n\n${text}"
    ),
    (Language.THAI, Language.VIETNAMESE): (
        "The following text was transcribed using a TTS model and may contain transcription errors or formatting issues. "
        "First, correct any transcription errors, remove unnecessary filler words, repetitions, or formatting inconsistencies. "
        "Then, translate the cleaned text accurately from Thai to Vietnamese. "
        "Provide only the final translated text:\n\n${text}"
    ),
}
