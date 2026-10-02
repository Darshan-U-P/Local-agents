from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


@dataclass
class SlideState:
    slide_number: int
    status: str = "pending"
    error: str | None = None


@dataclass
class GenerationSession:
    session_id: str
    topic: str
    output_dir: str
    status: str = "created"
    slides: list[SlideState] = field(default_factory=list)

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def __post_init__(self):
        self.output_path = Path(self.output_dir)
        self.output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ---------------------------------------------------------
    # SESSION FILE
    # ---------------------------------------------------------

    @property
    def session_file(self) -> Path:
        return self.output_path / "session.json"

    # ---------------------------------------------------------
    # SLIDES
    # ---------------------------------------------------------

    def initialize_slides(
        self,
        slide_count: int,
    ):
        self.slides = [
            SlideState(slide_number=i)
            for i in range(1, slide_count + 1)
        ]

        self.save()

    def update_slide(
        self,
        slide_number: int,
        status: str,
        error: str | None = None,
    ):
        for slide in self.slides:
            if slide.slide_number == slide_number:
                slide.status = status
                slide.error = error
                break
        else:
            raise ValueError(
                f"Slide {slide_number} does not exist."
            )

        self.save()

    # ---------------------------------------------------------
    # SESSION STATUS
    # ---------------------------------------------------------

    def set_status(self, status: str):
        self.status = status
        self.save()

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    def save(self):
        data = {
            "session_id": self.session_id,
            "topic": self.topic,
            "status": self.status,
            "created_at": self.created_at,
            "slides": [
                {
                    "slide_number": slide.slide_number,
                    "status": slide.status,
                    "error": slide.error,
                }
                for slide in self.slides
            ],
        }

        with self.session_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        session_file: str,
    ):
        path = Path(session_file)

        if not path.exists():
            raise FileNotFoundError(
                f"Session file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        session = cls(
            session_id=data["session_id"],
            topic=data["topic"],
            output_dir=str(path.parent),
            status=data["status"],
        )

        session.created_at = data["created_at"]

        session.slides = [
            SlideState(
                slide_number=slide["slide_number"],
                status=slide["status"],
                error=slide.get("error"),
            )
            for slide in data["slides"]
        ]

        return session