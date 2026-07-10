"""Central configuration + institution registry for the Educational Chatbot."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Institution:
    """Static metadata describing one institution the chatbot can serve."""

    code: str
    name: str
    full_name: str
    website: str
    allowed_hosts: tuple[str, ...]
    seed_urls: tuple[str, ...]
    contact_email: str = ""
    contact_phone: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


# --- Institution registry -------------------------------------------------
INSTITUTIONS: dict[str, Institution] = {
    "SRKI": Institution(
        code="SRKI",
        name="SRKI",
        full_name=(
            "Shree Ramkrishna Institute of Computer Education and Applied Sciences"
        ),
        website="https://www.srki.ac.in/",
        allowed_hosts=("srki.ac.in", "www.srki.ac.in"),
        seed_urls=(
            "https://www.srki.ac.in/",
            "https://www.srki.ac.in/pages/admission-corner/",
            "https://www.srki.ac.in/contact/",
            "https://www.srki.ac.in/pages/history/",
            "https://www.srki.ac.in/pages/courses-offered/",
        ),
        contact_email="info@srki.ac.in",
        contact_phone="7228018497",
        aliases=(
            "srki",
            "shree ramkrishna",
            "ramkrishna institute",
            "shree ramkrishna institute",
        ),
    ),
    "SU": Institution(
        code="SU",
        name="Sarvajanik University",
        full_name="Sarvajanik University",
        website="https://sarvajanikuniversity.ac.in",
        allowed_hosts=("sarvajanikuniversity.ac.in", "www.sarvajanikuniversity.ac.in"),
        seed_urls=(
            "https://sarvajanikuniversity.ac.in/",
            "https://sarvajanikuniversity.ac.in/admissions/",
            "https://sarvajanikuniversity.ac.in/academics/",
            "https://sarvajanikuniversity.ac.in/contact-us/",
        ),
        contact_email="info@sarvajanikuniversity.ac.in",
        contact_phone="",
        aliases=("su", "sarvajanik", "sarvajanik university"),
    ),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    active_institution: str = "SRKI"

    host: str = "127.0.0.1"
    port: int = 8000

    # Dataset source paths (used by scripts/prepare_data.py only)
    srki_dataset_a: str = r"E:\Final Datsets\Final_SRKI_dataset\Dataset_A_SRKI.csv"
    srki_dataset_b: str = r"E:\Final Datsets\Final_SRKI_dataset\Dataset_B_SRKI.csv"
    su_dataset_a: str = r"E:\Final Datsets\Final_SU_Dataset\SU_final_250k_A.csv"
    su_dataset_b: str = r"E:\Final Datsets\Final_SU_Dataset\SU_final_250k_B.csv"

    # Intent model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    srki_intent_model: str = ""
    su_intent_model: str = ""
    intent_confidence_threshold: float = 0.45
    max_seq_length: int = 128

    # Generator
    use_generator: bool = False
    srki_generator_model: str = "google/flan-t5-base"
    su_generator_model: str = "google/flan-t5-base"
    generator_max_input_chars: int = 3000
    generator_max_new_tokens: int = 220

    # Web scraping
    web_scrape_enabled: bool = True
    web_cache_ttl_hours: int = 24
    web_max_pages: int = 40
    web_request_timeout: int = 15
    web_request_delay_sec: float = 0.4
    web_user_agent: str = "EduChatbot/1.0 (educational; +local)"

    # Out-of-domain handling
    domain_guard_enabled: bool = True

    # Retrieval: the datasets' `ideal_response` values are templated filler and
    # NOT reliable factual answers, so by default only real scraped web content
    # is used to ground answers. Set true only if your Dataset B has real answers.
    rag_include_dataset: bool = False

    # --- Derived helpers -------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def processed_dir(self) -> Path:
        return ROOT / "data" / "processed"

    @property
    def index_dir(self) -> Path:
        return ROOT / "data" / "index"

    @property
    def web_cache_dir(self) -> Path:
        return ROOT / "data" / "web_cache"

    @property
    def models_dir(self) -> Path:
        return ROOT / "models"

    def institution(self, code: str | None = None) -> Institution:
        code = (code or self.active_institution).upper()
        return INSTITUTIONS.get(code, INSTITUTIONS["SRKI"])

    def dataset_paths(self, code: str) -> tuple[str, str]:
        code = code.upper()
        if code == "SU":
            return self.su_dataset_a, self.su_dataset_b
        return self.srki_dataset_a, self.srki_dataset_b

    def intent_model_source(self, code: str) -> str:
        """Return a HF repo id or local dir for the fine-tuned intent model."""
        code = code.upper()
        configured = self.su_intent_model if code == "SU" else self.srki_intent_model
        if configured:
            return configured
        local = self.models_dir / f"{code.lower()}_intent"
        return str(local) if local.exists() else ""

    def generator_source(self, code: str) -> str:
        code = code.upper()
        configured = (
            self.su_generator_model if code == "SU" else self.srki_generator_model
        )
        return configured or "google/flan-t5-base"


settings = Settings()
