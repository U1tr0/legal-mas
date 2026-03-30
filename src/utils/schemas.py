from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class OutcomeLabel(str, Enum):
    fine = "fine"
    deprivation = "deprivation"
    warning = "warning"
    arrest = "arrest"
    termination = "termination"
    upheld = "upheld"
    changed = "changed"
    other = "other"


class ArticleRef(BaseModel):
    """Нормализованная ссылка на статью закона."""
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="Кодекс/закон, напр. 'КоАП РФ'")
    article: str = Field(..., description="Номер статьи, напр. '12.15'")
    part: Optional[str] = Field(None, description="Часть статьи, напр. '4'")
    point: Optional[str] = Field(None, description="Пункт, если есть")
    raw: str = Field(..., description="Как было написано в документе")


class Punishment(BaseModel):
    """Структурированное представление назначенного наказания."""
    model_config = ConfigDict(extra="forbid")

    fine_rub: Optional[int] = Field(None, description="Размер штрафа в рублях")
    deprivation_months: Optional[int] = Field(None, description="Срок лишения в месяцах")
    arrest_days: Optional[int] = Field(None, description="Срок ареста в днях")
    community_service_hours: Optional[int] = Field(None, description="Обязательные работы, часы")
    other: Optional[str] = Field(None, description="Если не удалось распарсить нормально")


class CaseDocument(BaseModel):
    """Единое представление дела в слое обработанных данных."""
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., description="Уникальный идентификатор дела/документа")
    source: Optional[str] = Field(None, description="Источник (сайт/реестр)")
    source_url: Optional[str] = Field(None, description="URL на документ")
    document_type: Optional[str] = Field(None, description="Решение/постановление/определение и т.п.")

    decision_date: Optional[date] = None
    court_name: Optional[str] = None
    region: Optional[str] = None
    instance: Optional[str] = Field(None, description="Первая/апелляция/кассация, если известно")

    articles_raw: list[str] = Field(default_factory=list)
    articles_norm: list[ArticleRef] = Field(default_factory=list)
    outcome_raw: Optional[str] = None
    outcome_norm: Optional[OutcomeLabel] = None
    punishment: Optional[Punishment] = None

    summary: str = Field(..., description="Краткая фабула (для поиска и объяснения)")
    established: Optional[str] = None
    court_position: Optional[str] = None
    resolution: Optional[str] = None
    full_text: Optional[str] = None

    language: Literal["ru"] = "ru"
    created_at: Optional[datetime] = None
    hash: Optional[str] = Field(None, description="Хеш/версия текста (опционально)")


class Passage(BaseModel):
    """Фрагмент документа для поиска по отдельным пассажам."""
    model_config = ConfigDict(extra="forbid")

    passage_id: str
    case_id: str
    field: Literal["summary", "established", "court_position", "resolution"]
    text: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
