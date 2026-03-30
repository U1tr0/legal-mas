# legal-mas

`legal-mas` — экспериментальный пайплайн для юридического анализа административных кейсов.

Система принимает пользовательский запрос на естественном языке, затем по шагам:
- строит план поиска;
- ищет релевантные статьи КоАП и судебные дела;
- формирует grounded-ответ;
- проверяет качество и groundedness ответа;
- принимает финальное решение через orchestrator.

## Архитектура

Текущий основной runtime-контур:

1. `LegalPlanner`
2. `LawGuidedRetriever`
3. `LegalAnswerGenerator`
4. `LegalAnswerVerifier`
5. `LegalDecisionPipeline`

Поток данных:

`query -> PlannerOutput -> retrieval_result -> GeneratedAnswer -> VerificationResult -> PipelineResult`

## Структура проекта

### `src/`

- [`src/planner`](legal-mas\src\planner)
  - planner-слой
  - превращает сырой запрос в `PlannerOutput`
  - отвечает за `normalized_query`, `fact_description`, `search_queries`, `retrieval_targets`

- [`src/retrieval`](legal-mas\src\retrieval)
  - retrieval-слой
  - содержит BM25-индексы и `LawGuidedRetriever`
  - ищет статьи КоАП и судебные дела
  - формирует `evidence_pack`

- [`src/generator`](legal-mas\src\generator)
  - generator-слой
  - строит юридический ответ на основе planner output и evidence
  - возвращает `GeneratedAnswer`

- [`src/verifier`](legal-mas\src\verifier)
  - verifier-слой
  - оценивает groundedness и достаточность ответа
  - возвращает `VerificationResult`

- [`src/orchestrator`](legal-mas\src\orchestrator)
  - orchestration-слой
  - управляет вызовами planner/retriever/generator/verifier
  - поддерживает ограниченные циклы `REWRITE` и `RETRIEVE_MORE`
  - возвращает `PipelineResult`

- [`src/llm`](legal-mas\src\llm)
  - абстракции и реализации LLM-клиентов
  - сейчас используются:
    - `DeepSeekClient`
    - `QwenLocalClient`

- [`src/tasks`](legal-mas\src\tasks)
  - вспомогательные task-модули
  - например, генерация summary для судебных дел

- [`src/law`](legal-mas\src\law)
  - вспомогательная обработка юридических источников

- [`src/utils`](legal-mas\src\utils)
  - общие утилиты и вспомогательные схемы

### `scripts/`

- [`scripts/scrape_vsrf_raw.py`](legal-mas\scripts\scrape_vsrf_raw.py)
  - скачивание raw судебных дел

- [`scripts/prepare_dataset.py`](legal-mas\scripts\prepare_dataset.py)
  - сбор processed dataset из raw данных

- [`scripts/generate_summaries.py`](legal-mas\scripts\generate_summaries.py)
  - генерация summary для дел через LLM

- [`scripts/prepare_case_index.py`](legal-mas\scripts\prepare_case_index.py)
  - подготовка `case_index.parquet`

- [`scripts/prepare_koap_articles.py`](legal-mas\scripts\prepare_koap_articles.py)
  - подготовка корпуса статей КоАП

- [`scripts/prepare_koap_index.py`](legal-mas\scripts\prepare_koap_index.py)
  - подготовка `koap_index.parquet`

- [`scripts/test_planner_and_retrieval.py`](legal-mas\scripts\test_planner_and_retrieval.py)
  - быстрый тест planner + retrieval

- [`scripts/test_full_pipeline_generation.py`](legal-mas\scripts\test_full_pipeline_generation.py)
  - planner + retrieval + generator

- [`scripts/test_full_pipeline_with_verifier.py`](legal-mas\scripts\test_full_pipeline_with_verifier.py)
  - planner + retrieval + generator + verifier

- [`scripts/run_pipeline.py`](legal-mas\scripts\run_pipeline.py)
  - основной CLI entrypoint полного pipeline

### `data/`

Ожидаемые основные артефакты:

- `data/cases_raw.jsonl`
- `data/processed/cases.parquet`
- `data/processed/cases_with_summaries.parquet`
- `data/processed/case_index.parquet`
- `data/processed/koap_index.parquet`
- `data/pdf_documents/...`

## Основные сущности

### Planner

Возвращает `PlannerOutput`:
- `normalized_query`
- `fact_description`
- `query_type`
- `domain`
- `retrieval_targets`
- `candidate_articles`
- `search_queries`

### Retriever

Возвращает `retrieval_result`:
- `law_results`
- `case_results`
- `candidate_article_numbers`
- `evidence_pack`

`evidence_pack` состоит из:
- `law_evidence`
- `case_evidence`

### Generator

Возвращает `GeneratedAnswer`:
- `question_summary`
- `applicable_laws`
- `relevant_cases`
- `legal_analysis`
- `risk_factors`
- `final_answer`
- `confidence`

### Verifier

Возвращает `VerificationResult`:
- `verdict`
- `is_grounded`
- `is_sufficient`
- `problems`
- `explanation`
- `rewrite_focus`
- `confidence`

### Orchestrator

Возвращает `PipelineResult`:
- `status`
- `final_answer`
- `planner_output`
- `retrieval_result`
- `generated_answer`
- `verification_result`
- `iterations`
- `stop_reason`
- `trace`

## Быстрый старт

### 1. Полный pipeline из CLI

```powershell
python.exe .\scripts\run_pipeline.py "что будет за отказ от медосвидетельствования"
```

### 2. Тест planner + retrieval

```powershell
python.exe .\scripts\test_planner_and_retrieval.py "что будет за отказ от медосвидетельствования"
```

## Сборка индексов

Типовая цепочка подготовки данных:

1. `scrape_vsrf_raw.py`
2. `prepare_dataset.py`
3. `generate_summaries.py`
4. `prepare_case_index.py`
5. `prepare_koap_articles.py`
6. `prepare_koap_index.py`