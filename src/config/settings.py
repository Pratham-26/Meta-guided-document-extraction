import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class _Settings:
    def __init__(self):
        self.data_dir = Path(os.environ.get("DATA_DIR", "./data"))
        self.configs_dir = Path("configs")
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")

    @property
    def categories_dir(self) -> Path:
        return self.data_dir / "categories"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def model_config_path(self) -> Path:
        return self.configs_dir / "model_config.json"

    def ensure_dirs(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.categories_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)


settings = _Settings()
