"""
config.py — AiDarwaza Central Configuration
=============================================
Single source of truth for ALL settings, feature flags, and upgrade paths.

HOW TO USE:
    from config import cfg

    cfg.paths.db_path
    cfg.vision.min_det_score
    cfg.features.pashto
    cfg.ollama.model
"""

import os
import sys
import platform
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Root — resolves to wherever this file lives (no hardcoded paths ever)
# ---------------------------------------------------------------------------
_ROOT_ENV = os.environ.get("DOORBELL_ROOT")
ROOT: Path = Path(_ROOT_ENV).resolve() if _ROOT_ENV else Path(__file__).parent.resolve()

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
IS_WINDOWS: bool = platform.system() == "Windows"
IS_LINUX:   bool = platform.system() == "Linux"
IS_RPI:     bool = IS_LINUX and Path("/sys/firmware/devicetree/base/model").exists()
PYTHON_VERSION: tuple = sys.version_info[:3]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# FEATURE FLAGS
# ---------------------------------------------------------------------------
@dataclass
class FeatureFlags:
    # Language
    urdu:   bool = True
    pashto: bool = field(default_factory=lambda: _env_bool("FEATURE_PASHTO", False))

    # Audio output
    prerecorded_audio: bool = True
    live_tts:          bool = field(default_factory=lambda: _env_bool("FEATURE_LIVE_TTS", False))
    tts_engine:        str  = field(default_factory=lambda: os.environ.get("TTS_ENGINE", "coqui"))

    # STT
    whisper_stt:     bool = True
    faster_whisper:  bool = field(default_factory=lambda: _env_bool("FEATURE_FASTER_WHISPER", False))

    # Vision
    anti_spoofing: bool = field(default_factory=lambda: _env_bool("FEATURE_ANTISPOOF", True))
    rtsp_camera:   bool = field(default_factory=lambda: _env_bool("FEATURE_RTSP", False))
    multi_face:    bool = field(default_factory=lambda: _env_bool("FEATURE_MULTI_FACE", False))

    # Intelligence
    ollama_fallback:    bool = field(default_factory=lambda: _env_bool("FEATURE_OLLAMA", True))
    cloud_llm:          bool = field(default_factory=lambda: _env_bool("FEATURE_CLOUD_LLM", False))
    cloud_llm_provider: str  = field(default_factory=lambda: os.environ.get("CLOUD_LLM_PROVIDER", "anthropic"))

    # Notifications
    telegram_notify:  bool = field(default_factory=lambda: _env_bool("FEATURE_TELEGRAM", False))
    whatsapp_notify:  bool = field(default_factory=lambda: _env_bool("FEATURE_WHATSAPP", False))
    email_notify:     bool = field(default_factory=lambda: _env_bool("FEATURE_EMAIL", False))

    # Storage / Cloud
    cloud_sync: bool = field(default_factory=lambda: _env_bool("FEATURE_CLOUD_SYNC", False))
    s3_backup:  bool = field(default_factory=lambda: _env_bool("FEATURE_S3", False))

    # Hardware
    gpio_button: bool = field(default_factory=lambda: _env_bool("FEATURE_GPIO", IS_RPI))
    gpio_led:    bool = field(default_factory=lambda: _env_bool("FEATURE_GPIO_LED", IS_RPI))

    # Interface
    web_dashboard: bool = field(default_factory=lambda: _env_bool("FEATURE_WEB_DASHBOARD", False))

    # Developer tools
    debug_overlay:     bool = field(default_factory=lambda: _env_bool("FEATURE_DEBUG_OVERLAY", False))
    save_debug_frames: bool = field(default_factory=lambda: _env_bool("FEATURE_SAVE_FRAMES", False))
    dry_run:           bool = field(default_factory=lambda: _env_bool("DOORBELL_DRY_RUN", False))


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------
class Language(Enum):
    URDU    = "ur"
    PASHTO  = "ps"
    ENGLISH = "en"


@dataclass
class LanguageConfig:
    default_language:  Language = Language.URDU
    auto_detect:       bool     = False
    fallback_language: Language = Language.URDU

    whisper_models: dict = field(default_factory=lambda: {
        Language.URDU:    "small" if not IS_RPI else "tiny",
        Language.PASHTO:  "small",
        Language.ENGLISH: "base",
    })

    audio_suffixes: dict = field(default_factory=lambda: {
        Language.URDU:    "_ur.wav",
        Language.PASHTO:  "_ps.wav",
        Language.ENGLISH: "_en.wav",
    })


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass
class Paths:
    root:             Path = field(default_factory=lambda: ROOT)
    data:             Path = field(default_factory=lambda: ROOT / "data")
    faces_dir:        Path = field(default_factory=lambda: ROOT / "data" / "faces")
    audio_dir:        Path = field(default_factory=lambda: ROOT / "data" / "audio")
    responses_dir:    Path = field(default_factory=lambda: ROOT / "data" / "audio" / "responses")
    prompts_dir:      Path = field(default_factory=lambda: ROOT / "data" / "audio" / "prompts")
    tts_cache_dir:    Path = field(default_factory=lambda: ROOT / "data" / "audio" / "tts_cache")
    logs_dir:         Path = field(default_factory=lambda: ROOT / "data" / "logs")
    debug_frames_dir: Path = field(default_factory=lambda: ROOT / "data" / "debug_frames")

    db_path:          Path = field(default_factory=lambda: ROOT / "data" / "doorbell.db")
    faiss_index_path: Path = field(default_factory=lambda: ROOT / "data" / "faiss.index")

    models_dir:         Path = field(default_factory=lambda: ROOT / "models")
    antispoof_json:     Path = field(default_factory=lambda: ROOT / "models" / "antispoofing_model.json")
    antispoof_weights:  Path = field(default_factory=lambda: ROOT / "models" / "antispoofing_model.h5")
    intent_model_path:  Path = field(default_factory=lambda: ROOT / "nlp" / "models" / "intent_model.pkl")

    tts_model_dir:   Path = field(default_factory=lambda: ROOT / "models" / "tts")
    ota_staging_dir: Path = field(default_factory=lambda: ROOT / "models" / "ota_staging")
    plugins_dir:     Path = field(default_factory=lambda: ROOT / "plugins")

    def ensure_all(self) -> None:
        """Create all required directories. Safe to call on every startup."""
        dirs = [
            self.data, self.faces_dir, self.audio_dir, self.responses_dir,
            self.prompts_dir, self.tts_cache_dir, self.logs_dir,
            self.debug_frames_dir, self.models_dir, self.tts_model_dir,
            self.ota_staging_dir, self.plugins_dir,
            ROOT / "nlp" / "models",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def validate_models(self) -> list:
        required = [
            (self.antispoof_json,    "Anti-spoofing model architecture (.json)"),
            (self.antispoof_weights, "Anti-spoofing model weights (.h5)"),
            (self.intent_model_path, "Intent classifier (.pkl)"),
        ]
        return [
            f"  MISSING: {label}\n          → {path}"
            for path, label in required if not path.exists()
        ]

    def validate_audio_responses(self) -> list:
        wavs = [
            "ask_name_ur.wav",           "ask_name_retry_ur.wav",
            "ask_intent_ur.wav",         "ask_intent_retry_ur.wav",
            "greeting_known_ur.wav",     "greeting_unknown_ur.wav",
            "parcel_instruction_ur.wav", "emergency_ur.wav",
            "fallback_low_confidence_ur.wav", "fallback_no_name_ur.wav",
        ]
        return [
            f"  MISSING audio: {w}"
            for w in wavs if not (self.responses_dir / w).exists()
        ]


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
@dataclass
class VisionConfig:
    insightface_model:       str   = "buffalo_l"
    det_size:                tuple = (640, 640)
    min_det_score:           float = field(default_factory=lambda: float(os.environ.get("VISION_MIN_DET_SCORE", "0.45")))
    blur_threshold:          float = 15.0
    min_eye_distance:        int   = 10
    min_mouth_distance:      int   = 6
    min_face_area:           int   = 2500
    stability_seconds:       float = 1.5
    disappear_seconds:       float = 2.0
    antispoof_input_size:    tuple = (224, 224)
    antispoof_real_threshold:float = 0.5
    camera_index:            int   = field(default_factory=lambda: int(os.environ.get("CAMERA_INDEX", "0")))
    rtsp_url:                str   = field(default_factory=lambda: os.environ.get("RTSP_URL", ""))


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
@dataclass
class RecognitionConfig:
    embedding_dim:      int   = 512
    match_threshold:    float = field(default_factory=lambda: float(os.environ.get("FAISS_THRESHOLD", "0.80")))
    search_k:           int   = 1
    return_confidence:  bool  = True
    cross_zone_matching:bool  = False


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
@dataclass
class AudioConfig:
    sample_rate:                  int   = 16000
    channels:                     int   = 1
    record_seconds:               int   = field(default_factory=lambda: int(os.environ.get("AUDIO_RECORD_SECONDS", "6")))
    min_audio_duration:           float = 1.0
    min_rms_volume:               int   = 200
    whisper_language:             str   = "ur"
    whisper_fp16:                 bool  = False
    faster_whisper_compute_type:  str   = "int8"
    speaker_diarization:          bool  = False

    @property
    def whisper_model(self) -> str:
        override = os.environ.get("WHISPER_MODEL", "")
        if override:
            return override
        return "tiny" if IS_RPI else "small"


# ---------------------------------------------------------------------------
# NLP
# ---------------------------------------------------------------------------
@dataclass
class NLPConfig:
    intent_confidence_threshold: float = field(default_factory=lambda: float(os.environ.get("INTENT_THRESHOLD", "0.65")))
    ml_confidence_threshold:     float = 0.60
    max_dialogue_retries:        int   = 2
    name_max_words:              int   = 3
    name_max_chars:              int   = 20
    name_min_char_length:        int   = 2
    multi_intent:                bool  = False
    conversation_memory:         bool  = False

    urdu_stopwords: frozenset = field(default_factory=lambda: frozenset({
        "ہے", "ہوں", "تھا", "تھی", "کر", "کا", "کی", "کے",
        "میں", "آپ", "ہم", "یہ", "وہ", "اور", "سے", "نے",
        "کو", "پر", "تو", "بھی", "جو", "اس", "ان", "اک",
        "ایک", "کہ", "لئے", "لیے", "مجھے", "مجھ", "تم",
    }))

    pashto_stopwords: frozenset = field(default_factory=lambda: frozenset({
        "دی", "چې", "د", "او", "چه", "وه", "ده", "یم", "یې",
    }))


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
@dataclass
class OllamaConfig:
    base_url:              str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    model:                 str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct"))
    temperature:         float = 0.15
    top_p:               float = 0.9
    timeout_seconds: int = int(os.environ.get("OLLAMA_TIMEOUT", "30"))
    intent_model:          str = field(default_factory=lambda: os.environ.get("OLLAMA_INTENT_MODEL", "qwen2.5:3b-instruct"))
    name_extraction_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_NAME_MODEL",   "qwen2.5:3b-instruct"))
    summary_model:         str = field(default_factory=lambda: os.environ.get("OLLAMA_SUMMARY_MODEL","qwen2.5:7b-instruct"))

    @property
    def generate_url(self) -> str:
        return f"{self.base_url}/api/generate"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/api/tags"


# ---------------------------------------------------------------------------
# Cloud LLM fallback (Phase 3)
# ---------------------------------------------------------------------------
@dataclass
class CloudLLMConfig:
    provider:          str = field(default_factory=lambda: os.environ.get("CLOUD_LLM_PROVIDER", "anthropic"))
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    anthropic_model:   str = "claude-haiku-4-5-20251001"
    openai_api_key:    str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_model:      str = "gpt-4o-mini"
    max_tokens:        int = 200
    timeout_seconds:   int = 15


# ---------------------------------------------------------------------------
# Notifications (Phase 2)
# ---------------------------------------------------------------------------
@dataclass
class NotificationConfig:
    telegram_bot_token:  str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id:    str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))
    whatsapp_account_sid:str = field(default_factory=lambda: os.environ.get("TWILIO_ACCOUNT_SID", ""))
    whatsapp_auth_token: str = field(default_factory=lambda: os.environ.get("TWILIO_AUTH_TOKEN", ""))
    whatsapp_from:       str = field(default_factory=lambda: os.environ.get("WHATSAPP_FROM", ""))
    whatsapp_to:         str = field(default_factory=lambda: os.environ.get("WHATSAPP_TO", ""))
    smtp_host:           str = field(default_factory=lambda: os.environ.get("SMTP_HOST", "smtp.gmail.com"))
    smtp_port:           int = field(default_factory=lambda: int(os.environ.get("SMTP_PORT", "587")))
    smtp_user:           str = field(default_factory=lambda: os.environ.get("SMTP_USER", ""))
    smtp_password:       str = field(default_factory=lambda: os.environ.get("SMTP_PASSWORD", ""))
    notify_email:        str = field(default_factory=lambda: os.environ.get("NOTIFY_EMAIL", ""))
    include_face_crop:   bool = True
    include_transcript:  bool = True
    include_intent:      bool = True


# ---------------------------------------------------------------------------
# Cloud Sync (Phase 3)
# ---------------------------------------------------------------------------
@dataclass
class CloudConfig:
    sync_url:              str  = field(default_factory=lambda: os.environ.get("CLOUD_SYNC_URL", ""))
    api_key:               str  = field(default_factory=lambda: os.environ.get("CLOUD_API_KEY", ""))
    s3_bucket:             str  = field(default_factory=lambda: os.environ.get("S3_BUCKET", ""))
    s3_access_key:         str  = field(default_factory=lambda: os.environ.get("S3_ACCESS_KEY", ""))
    s3_secret_key:         str  = field(default_factory=lambda: os.environ.get("S3_SECRET_KEY", ""))
    s3_endpoint:           str  = field(default_factory=lambda: os.environ.get("S3_ENDPOINT", ""))
    sync_events:           bool = True
    sync_face_crops:       bool = True
    sync_audio_recordings: bool = False
    retention_days:        int  = field(default_factory=lambda: int(os.environ.get("RETENTION_DAYS", "30")))


# ---------------------------------------------------------------------------
# TTS (Phase 2)
# ---------------------------------------------------------------------------
@dataclass
class TTSConfig:
    engine:               str  = field(default_factory=lambda: os.environ.get("TTS_ENGINE", "coqui"))
    urdu_voice:           str  = field(default_factory=lambda: os.environ.get("TTS_URDU_VOICE", ""))
    pashto_voice:         str  = field(default_factory=lambda: os.environ.get("TTS_PASHTO_VOICE", ""))
    speed:                float = 1.0
    cache_generated_audio: bool = True
    cache_dir:            Path = field(default_factory=lambda: ROOT / "data" / "audio" / "tts_cache")


# ---------------------------------------------------------------------------
# Hardware / GPIO
# ---------------------------------------------------------------------------
@dataclass
class HardwareConfig:
    enroll_button_pin:    int   = field(default_factory=lambda: int(os.environ.get("GPIO_ENROLL_PIN", "17")))
    status_led_pin:       int   = field(default_factory=lambda: int(os.environ.get("GPIO_LED_PIN", "27")))
    doorbell_trigger_pin: int   = field(default_factory=lambda: int(os.environ.get("GPIO_TRIGGER_PIN", "22")))
    door_relay_pin:       int   = field(default_factory=lambda: int(os.environ.get("GPIO_RELAY_PIN", "0")))
    enroll_hold_seconds:  float = 3.0
    button_debounce_ms:   int   = 200
    door_unlock_seconds:  float = 3.0
    led_blink_detecting:  float = 0.5
    led_solid_known:      float = 0.0
    led_blink_unknown:    float = 0.2


# ---------------------------------------------------------------------------
# Web Dashboard (Phase 3)
# ---------------------------------------------------------------------------
@dataclass
class DashboardConfig:
    host:             str  = field(default_factory=lambda: os.environ.get("DASHBOARD_HOST", "0.0.0.0"))
    port:             int  = field(default_factory=lambda: int(os.environ.get("WEB_PORT", "8080")))
    secret_key:       str  = field(default_factory=lambda: os.environ.get("DASHBOARD_SECRET_KEY", "change-me-in-production"))
    enable_auth:      bool = True
    admin_username:   str  = field(default_factory=lambda: os.environ.get("DASHBOARD_USER", "admin"))
    admin_password:   str  = field(default_factory=lambda: os.environ.get("DASHBOARD_PASS", ""))
    enable_live_feed: bool = False


# ---------------------------------------------------------------------------
# System / Runtime
# ---------------------------------------------------------------------------
@dataclass
class SystemConfig:
    cooldown_seconds:        int   = field(default_factory=lambda: int(os.environ.get("DOORBELL_COOLDOWN", "30")))
    log_level:               str   = field(default_factory=lambda: os.environ.get("DOORBELL_LOG_LEVEL", "INFO"))
    log_to_file:             bool  = True
    log_max_bytes:           int   = 5 * 1024 * 1024
    log_backup_count:        int   = 3
    show_preview_window:     bool  = field(default_factory=lambda: os.environ.get("DOORBELL_HEADLESS", "0") != "1")
    preview_window_title:    str   = "AiDarwaza — AI Smart Doorbell"
    preview_scale:           float = 1.0
    max_face_crops:          int   = field(default_factory=lambda: int(os.environ.get("MAX_FACE_CROPS", "1000")))
    max_audio_recordings:    int   = field(default_factory=lambda: int(os.environ.get("MAX_AUDIO_FILES", "500")))
    cleanup_older_than_days: int   = field(default_factory=lambda: int(os.environ.get("CLEANUP_DAYS", "30")))
    frame_skip:              int   = field(default_factory=lambda: int(os.environ.get("FRAME_SKIP", "4")))
    max_fps:                 int   = field(default_factory=lambda: int(os.environ.get("MAX_FPS", "30")))


# ---------------------------------------------------------------------------
# Plugin Registry
# ---------------------------------------------------------------------------
@dataclass
class PluginRegistry:
    enabled_plugins:        list  = field(default_factory=lambda: [
        p for p in os.environ.get("ENABLED_PLUGINS", "").split(",") if p
    ])
    plugin_timeout_seconds: float = 2.0


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------
class Config:
    def __init__(self):
        self.features      = FeatureFlags()
        self.paths         = Paths()
        self.language      = LanguageConfig()
        self.vision        = VisionConfig()
        self.recognition   = RecognitionConfig()
        self.audio         = AudioConfig()
        self.nlp           = NLPConfig()
        self.ollama        = OllamaConfig()
        self.cloud_llm     = CloudLLMConfig()
        self.notifications = NotificationConfig()
        self.cloud         = CloudConfig()
        self.tts           = TTSConfig()
        self.hardware      = HardwareConfig()
        self.dashboard     = DashboardConfig()
        self.system        = SystemConfig()
        self.plugins       = PluginRegistry()

        self.paths.ensure_all()
        self._configure_logging()

    def _configure_logging(self) -> None:
        from logging.handlers import RotatingFileHandler
        level  = getattr(logging, self.system.log_level.upper(), logging.INFO)
        fmt    = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
        datefmt= "%Y-%m-%d %H:%M:%S"
        root   = logging.getLogger()
        root.setLevel(level)
        if root.handlers:
            return
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(ch)
        if self.system.log_to_file:
            fh = RotatingFileHandler(
                self.paths.logs_dir / "doorbell.log",
                maxBytes=self.system.log_max_bytes,
                backupCount=self.system.log_backup_count,
                encoding="utf-8",
            )
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt, datefmt))
            root.addHandler(fh)

    def validate(self, strict: bool = False) -> bool:
        logger  = logging.getLogger("config.validate")
        all_ok  = True
        missing = self.paths.validate_models() + self.paths.validate_audio_responses()
        for msg in missing:
            all_ok = False
            logger.warning(msg)
        if self.features.web_dashboard and not self.dashboard.admin_password:
            logger.warning("Web dashboard enabled but DASHBOARD_PASS not set in .env!")
        if self.features.telegram_notify and not self.notifications.telegram_bot_token:
            logger.warning("Telegram enabled but TELEGRAM_BOT_TOKEN not set in .env!")
        if not all_ok:
            if strict:
                raise RuntimeError("Pre-flight validation failed. Check warnings above.")
            logger.warning("Running in degraded mode — missing files listed above.")
        else:
            logger.info("Pre-flight validation passed. All required files found.")
        return all_ok

    def active_features(self) -> list:
        return [k for k, v in self.features.__dict__.items() if isinstance(v, bool) and v]

    def summary(self) -> str:
        active = ", ".join(self.active_features()) or "core only"
        return "\n".join([
            "=" * 62,
            "  AiDarwaza — Configuration Summary",
            "=" * 62,
            f"  Platform     : {platform.system()} ({'RPi' if IS_RPI else 'PC/Server'})",
            f"  Python       : {'.'.join(str(v) for v in PYTHON_VERSION)}",
            f"  Root         : {ROOT}",
            f"  Database     : {self.paths.db_path}",
            "",
            "  [Active Features]",
            f"  {active}",
            "",
            f"  [Vision]      {self.vision.insightface_model}  anti-spoof={'ON' if self.features.anti_spoofing else 'OFF'}",
            f"  [Languages]   Urdu {'+ Pashto' if self.features.pashto else ''}",
            f"  [STT]         Whisper {self.audio.whisper_model}",
            f"  [TTS]         {'Live (' + self.tts.engine + ')' if self.features.live_tts else 'Pre-recorded WAV'}",
            f"  [LLM]         {self.ollama.model} @ {self.ollama.base_url}",
            f"  [Notify]      Telegram={'ON' if self.features.telegram_notify else 'off'}  WA={'ON' if self.features.whatsapp_notify else 'off'}",
            f"  [System]      cooldown={self.system.cooldown_seconds}s  headless={not self.system.show_preview_window}",
            f"  [Plugins]     {self.plugins.enabled_plugins or 'none'}",
            "=" * 62,
        ])


# ---------------------------------------------------------------------------
# Singleton — import this everywhere, never instantiate Config() directly
# ---------------------------------------------------------------------------
cfg = Config()
