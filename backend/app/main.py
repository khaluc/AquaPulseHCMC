"""AquaPulse - environmental screening, not AI diagnosis."""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import httpx
import psycopg
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from app import (
    alerts,
    community,
    dashboard,
    fhir,
    i18n,
    metrics,
    notifications,
    oneaqua,
    press,
    repository,
    translator,
    verification,
    waterways,
)
from app.auth import Reviewer, current_reviewer, volunteer_ids
from app.db import load_default_database
from app.detector import CLASS_INFO, InvalidImageError, load_default_detector, load_image
from app.domain import PRODUCT_NAME, scope_summary
from app.fusion import ObservationGroup, ReportAnswers, ReporterLevel
from app.pipeline import DETECTION_DISCLAIMER, WATER_NOTE, PipelineResult, rescore, run_pipeline
from app.quality import QualityReport, check_image
from app.review import ReviewRoute
from app.risk import V_WATER
from app.storage import photo_dir, save_photo
from app.trends import classify
from app.vector_watch import CATALOG, POSITIVE_AT
from app.vector_watch import catalog as vector_watch_catalog
from app.vision import load_default_vision
from app.water import load_default_segmenter

STATIC_DIR = Path(__file__).resolve().parent / "static"
log = logging.getLogger("uvicorn.error")  # uvicorn prints its own loggers
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

# Flow step G for observations. A 500 m hotspot is meaningless from a fix that is off by
# more than half of that; a photo "taken" in the future means a wrong phone clock.
MAX_LOCATION_ACCURACY_M = 250
MAX_CLOCK_SKEW = timedelta(minutes=10)

VISION_NOTE = (
    "Model thị giác chỉ mô tả bối cảnh và giải thích. Nó không quyết định mức rủi ro; "
    "phần đó do rule engine và người xác minh."
)
PHOTO_PRIVACY = {
    False: "Ảnh được phân tích ngay trên máy chủ này, không gửi ra ngoài.",
    True: "Ảnh được gửi tới dịch vụ bên ngoài để phân tích.",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "detector"):
        app.state.detector = load_default_detector()
    if not hasattr(app.state, "water"):
        app.state.water = load_default_segmenter()
    if not hasattr(app.state, "vision"):
        app.state.vision = load_default_vision()
    if not hasattr(app.state, "database"):
        app.state.database = await run_in_threadpool(load_default_database)
    if app.state.database is not None:
        await run_in_threadpool(_load_segments, app.state.database)
    press_task = None
    press_hours = float(os.environ.get("AQUAPULSE_PRESS_SYNC_HOURS") or 24)
    if app.state.database is not None and press_hours > 0:
        press_task = asyncio.create_task(sync_press_forever(app.state.database, press_hours))
    oah_task = None
    hours = float(os.environ.get("AQUAPULSE_OAH_SYNC_HOURS") or 24)  # 0 turns it off
    if app.state.database is not None and hours > 0:
        oah_task = asyncio.create_task(sync_oneaqua_forever(app.state.database, hours))
    detector = getattr(app.state, "detector", None)
    if detector is not None:
        # The first run costs ~0.6 s per thread, so warm up in the threadpool that
        # /detect uses, not on the event loop thread.
        await run_in_threadpool(detector.detect, Image.new("RGB", (64, 64)), 1.0)
    water = getattr(app.state, "water", None)
    if water is not None:
        await run_in_threadpool(water.segment, Image.new("RGB", (64, 64)))
    yield
    for task in (oah_task, press_task):
        if task is not None:
            task.cancel()


async def sync_press_forever(database, hours: float) -> None:
    """Refresh the press index (Google News RSS) at start-up and then every `hours`."""
    while True:
        try:
            counts = await run_in_threadpool(press.sync, database)
            log.info("Press sync: %s", counts)
        except (httpx.HTTPError, ValueError, KeyError, psycopg.Error) as exc:
            log.warning("Press sync failed: %s", exc)
        await asyncio.sleep(hours * 3600)


async def sync_oneaqua_forever(database, hours: float) -> None:
    """Refresh the OneAquaHealth data at start-up and then every `hours`; a failed run keeps
    what is already stored."""
    while True:
        try:
            counts = await run_in_threadpool(oneaqua.sync, database)
            log.info("OneAquaHealth sync: %s", counts)
        except (httpx.HTTPError, ValueError, KeyError, psycopg.Error) as exc:
            log.warning("OneAquaHealth sync failed: %s", exc)
        await asyncio.sleep(hours * 3600)


def _load_segments(database) -> None:
    with database.connect() as conn:
        waterways.load_segments(conn)


app = FastAPI(title=f"{PRODUCT_NAME} API", version="0.1.0", lifespan=lifespan)
# Shared page assets (theme.css). The pages themselves have their own routes below.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# The API answers in Vietnamese, the language people report in. `?lang=en` translates the
# reader-facing strings of a JSON answer on the way out (app/i18n.py); the pages append it
# when the visitor picks EN. FHIR resources are left alone: they go to another system, not
# to a reader, and their text is part of the record.
NO_TRANSLATE = ("/fhir/",)


@app.middleware("http")
async def language(request: Request, call_next):
    response = await call_next(request)
    if (response.headers.get("content-type", "").startswith("text/html")
            or request.url.path == "/static/i18n.js"):
        response.headers["Cache-Control"] = "no-cache"
    lang = i18n.normalize(request.query_params.get("lang"))
    bilingual = request.headers.get("X-AquaPulse-Languages") == "vi,en"
    if (request.url.path.startswith(NO_TRANSLATE)
            or not response.headers.get("content-type", "").startswith("application/json")):
        return response
    response.headers.append("Vary", "X-AquaPulse-Languages")
    if lang == i18n.DEFAULT and not bilingual:
        return response
    body = b"".join([chunk async for chunk in response.body_iterator])
    try:
        payload = json.loads(body)
    except ValueError:
        return Response(body, status_code=response.status_code, headers=dict(response.headers))
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    # The report form switches language without re-running models or saving another report.
    # Both views come from the same result, so scores, identifiers and evidence stay the same.
    localized = {"vi": payload, "en": i18n.localize(payload, "en")} if bilingual else i18n.localize(payload, lang)
    return JSONResponse(localized, status_code=response.status_code, headers=headers)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "aquapulse"}


@app.get("/scope")
def scope() -> dict[str, object]:
    return scope_summary()


def page(name: str) -> FileResponse:
    return FileResponse(STATIC_DIR / name, media_type="text/html; charset=utf-8")


@app.get("/", include_in_schema=False)
def home_page() -> FileResponse:
    return page("home.html")


@app.get("/report", include_in_schema=False)
def report_page() -> FileResponse:
    return page("demo.html")


@app.get("/passport", include_in_schema=False)
def passport_page() -> FileResponse:
    return page("passport.html")


@app.get("/tasks-board", include_in_schema=False)
def tasks_page() -> FileResponse:
    return page("tasks.html")


@app.get("/config")
def public_config() -> dict[str, object]:
    """What the web pages need to know; nothing secret."""
    return {
        "map_tile_url": os.environ.get("MAP_TILE_URL", "https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png"),
        "map_attribution": os.environ.get(
            "MAP_ATTRIBUTION", "&copy; OpenStreetMap contributors, tiles: Humanitarian OSM Team"),
    }


def require_detector(request: Request):
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Chưa có model. Export ml/runs/n_clean/weights/best.onnx hoặc đặt AQUAPULSE_DETECTOR_MODEL.",
        )
    return detector


def require_database(request: Request):
    database = getattr(request.app.state, "database", None)
    if database is None:
        raise HTTPException(
            status_code=503,
            detail="Chưa kết nối database. Chạy `docker compose up -d db` hoặc đặt DATABASE_URL.",
        )
    return database


@app.get("/detector")
def detector_info(request: Request) -> dict[str, object]:
    detector = require_detector(request)
    water = getattr(request.app.state, "water", None)
    vision = getattr(request.app.state, "vision", None)
    return {
        "model": detector.model_path.name,
        "run": detector.model_path.parent.parent.name,
        "input_size": detector.imgsz,
        "alert_threshold": detector.threshold,
        "license": detector.license,
        "database": getattr(request.app.state, "database", None) is not None,
        "vision_model": None if vision is None else {
            "provider": vision.provider,
            "model": vision.model,
            "note": VISION_NOTE,
            "privacy": PHOTO_PRIVACY[vision.sends_photo_off_device],
        },
        "water_model": None if water is None else {
            "model": water.model_path.name,
            "run": water.model_path.parent.parent.name,
            "confidence": water.confidence,
            "note": WATER_NOTE,
        },
        "classes": [
            {"id": class_id, "name": name, **CLASS_INFO.get(name, {"label": name, "validated": False, "note": ""})}
            for class_id, name in sorted(detector.names.items())
        ],
        "disclaimer": DETECTION_DISCLAIMER,
    }


def parse_answers(answers: str | None) -> ReportAnswers:
    try:
        return ReportAnswers.model_validate_json(answers) if answers else ReportAnswers()
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Câu trả lời nhanh không hợp lệ: {exc}") from exc


async def read_photo(image: UploadFile) -> Image.Image:
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ảnh quá lớn, tối đa 15 MB.")
    try:
        return await run_in_threadpool(load_image, data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def quality_gate(picture: Image.Image, skip: bool = False) -> QualityReport:
    report = await run_in_threadpool(check_image, picture)
    if not report.usable and not skip:
        # Flow step L: an unusable photo goes back to the reporter instead of the model.
        raise HTTPException(
            status_code=422,
            detail={"message": "Ảnh chưa dùng được, cần chụp lại.", "quality": report.as_dict()},
        )
    return report


@app.post("/detect")
async def detect(
    request: Request,
    image: UploadFile = File(...),
    answers: str | None = Form(None, description="Câu trả lời nhanh, JSON theo ReportAnswers"),
    conf_floor: float = Query(0.05, ge=0.01, le=1.0),
    skip_quality: bool = Query(False),
    with_water: bool = Query(True),
    with_vision: bool = Query(False),
    with_detector: bool = Query(True, description="false: chỉ model mô tả, không chạy model vật chứa"),
    lang: str = Query(i18n.DEFAULT, description="vi | en, also the language the scene model answers in"),
) -> dict[str, object]:
    """Score a photo without storing anything."""
    if with_detector:
        require_detector(request)
    report_answers = parse_answers(answers)
    picture = await read_photo(image)
    report = await quality_gate(picture, skip_quality)
    result = await run_pipeline(
        request.app.state, picture, report_answers, report,
        conf_floor=conf_floor, with_water=with_water, with_vision=with_vision, with_detector=with_detector,
        lang=i18n.normalize(lang),
    )
    return result.payload


def validate_location_and_time(
    latitude: float, longitude: float, accuracy: float | None, captured_at: datetime | None
) -> datetime:
    if latitude == 0 and longitude == 0:
        raise HTTPException(status_code=422, detail="GPS chưa bắt được vị trí (0, 0). Chờ định vị rồi gửi lại.")
    if accuracy is not None and accuracy > MAX_LOCATION_ACCURACY_M:
        raise HTTPException(
            status_code=422,
            detail=f"Vị trí chưa đủ chính xác (±{accuracy:.0f} m). Cần dưới {MAX_LOCATION_ACCURACY_M} m.",
        )
    now = datetime.now(UTC)
    if captured_at is None:
        return now
    if captured_at.tzinfo is None:
        # A naive time is exactly the "+07:00 dropped" bug from section 5; refuse to guess.
        raise HTTPException(status_code=422, detail="captured_at phải kèm múi giờ, ví dụ 2026-09-17T15:30:00+07:00.")
    if captured_at > now + MAX_CLOCK_SKEW:
        raise HTTPException(status_code=422, detail="Thời điểm chụp ở tương lai; kiểm tra đồng hồ điện thoại.")
    return captured_at


def observation_categories(result: PipelineResult) -> list[str]:
    """What a report is about, for grouping into hotspots ('cùng loại dấu hiệu')."""
    components = result.assessment.components
    watch = result.vector_watch
    categories = []
    # The Vector Watch domain, when the group's own rule found something (or the group
    # is record-only): a "no standing water here" report must not join a mosquito hotspot.
    if watch.group_score is None or watch.group_score >= POSITIVE_AT:
        categories.append(watch.spec.domain)
    # Water in a container, or larvae. Not for river reports: a bottle floating in a canal
    # must not turn an oil-sheen report into a mosquito alert; mosquitoes have their own group.
    if ((components["V"].value or 0) >= V_WATER and "vector" not in categories
            and watch.group is not ObservationGroup.RIVER_HEALTH):
        categories.append("vector")
    if (components["E"].value or 0) >= 0.6:
        categories.append("environment")
    return categories


@app.post("/observations", status_code=201)
async def create_observation(
    request: Request,
    image: UploadFile = File(...),
    latitude: float = Form(..., ge=-90, le=90),
    longitude: float = Form(..., ge=-180, le=180),
    location_accuracy_m: float | None = Form(None, ge=0),
    captured_at: datetime | None = Form(None, description="ISO 8601 kèm múi giờ"),
    reporter_id: str | None = Form(None, max_length=64),
    answers: str | None = Form(None, description="Câu trả lời nhanh, JSON theo ReportAnswers"),
    with_vision: bool = Query(False),
    with_detector: bool = Query(True, description="false: chỉ model mô tả, không chạy model vật chứa"),
    lang: str = Query(i18n.DEFAULT, description="vi | en, also the language the scene model answers in"),
) -> dict[str, object]:
    """Flow steps F-Q, then J: validate, check quality, score, and only then store.

    The plan's flow stores the photo (J) before the quality check (K). Here an
    unusable photo is sent back without ever being written, so rejected pictures of
    people's homes do not pile up on disk.
    """
    if with_detector:
        require_detector(request)
    database = require_database(request)
    observed_at = validate_location_and_time(latitude, longitude, location_accuracy_m, captured_at)
    report_answers = await with_server_reporter_level(database, parse_answers(answers), reporter_id)
    nearby = await nearby_reports(database, latitude, longitude, report_answers)
    picture = await read_photo(image)
    report = await quality_gate(picture)
    result = await run_pipeline(
        request.app.state, picture, report_answers, report,
        conf_floor=0.05, with_water=True, with_vision=with_vision, with_detector=with_detector,
        nearby_reports=nearby, lang=i18n.normalize(lang),
    )
    stored = await run_in_threadpool(save_photo, picture)
    item = scored_observation(
        result,
        reporter_id=reporter_id,
        latitude=latitude,
        longitude=longitude,
        location_accuracy_m=location_accuracy_m,
        observed_at=observed_at,
        image_path=stored.relative_path,
        image_sha256=stored.sha256,
        image_phash_hex=str(report.metrics["phash"]),
    )

    def insert():
        with database.connect() as conn:
            observation_id, duplicate_of = repository.insert_observation(conn, item)
            task_id = verification.queue(conn, observation_id, result.review) if result.review.decision == "verify" else None
            points = community.award_report(conn, reporter_id, observation_id, duplicate_of is not None)
            changes = alerts.refresh(conn) if duplicate_of is None else []
            segment = waterways.segment_for(conn, latitude, longitude)
            near_site = oneaqua.nearest_site(conn, latitude, longitude)
            return observation_id, duplicate_of, task_id, points, changes, segment, near_site

    try:
        (observation_id, duplicate_of, task_id, points, changes, segment,
         near_site) = await run_in_threadpool(insert)
    except psycopg.OperationalError as exc:
        raise HTTPException(status_code=503, detail="Mất kết nối database.") from exc
    return {
        "id": str(observation_id),
        "status": item.status,
        "verification_task_id": None if task_id is None else str(task_id),
        "duplicate_of": None if duplicate_of is None else str(duplicate_of),
        "categories": item.categories,
        "points_awarded": points,
        "alert_changes": [{**c, "alert_id": str(c["alert_id"])} for c in changes],
        "faces_blurred": stored.faces_blurred,
        "segment": segment,
        "oneaquahealth_site": near_site,
        **result.payload,
    }


async def nearby_reports(database, latitude: float, longitude: float, answers: ReportAnswers,
                         exclude_id: UUID | None = None) -> int:
    """Same-group reports nearby this week, for the Culex rule's 'repeated_reports'."""
    group = answers.observation_group.value if answers.observation_group else None

    def count():
        with database.connect() as conn:
            return repository.count_nearby_reports(conn, latitude, longitude, group, exclude_id)

    return await run_in_threadpool(count)


def scored_observation(result: PipelineResult, **fields) -> repository.NewObservation:
    answers = result.payload["answers"]
    watch = result.vector_watch
    return repository.NewObservation(
        **fields,
        observation_types=list(answers["observation_types"]),
        answers=answers,
        signals=[name for name, signal in result.fusion.signals.items() if signal.present],
        categories=observation_categories(result),
        risk=result.assessment.risk,
        confidence=result.assessment.confidence,
        level=result.assessment.level,
        needs_verification=result.review.decision == "verify",
        analysis=result.payload,
        status=result.review.status,
        review_reasons=result.review.reasons,
        expert_required=result.review.expert_required,
        risk_domain=watch.spec.domain,
        vector_group=watch.group.value,
        group_score=watch.group_score,
        vector_breeding_risk=watch.vector_breeding_risk,
        environmental_signal=watch.environmental_signal,
        human_contact_risk=watch.human_contact_risk,
        eco_score=watch.eco_score,
        observation_goal=None if answers.get("observation_goal") in (None, "unknown") else answers["observation_goal"],
    )


async def with_server_reporter_level(database, answers: ReportAnswers, reporter_id: str | None) -> ReportAnswers:
    """Replace any self-declared reporter level with the one the database supports."""

    def lookup():
        with database.connect() as conn:
            return repository.reporter_level(conn, reporter_id, volunteer_ids())

    # model_copy does not validate, so pass the enum itself rather than its string value.
    return answers.model_copy(update={"reporter_level": ReporterLevel(await run_in_threadpool(lookup))})


@app.post("/observations/{observation_id}/confirm")
async def confirm_observation(
    request: Request,
    observation_id: UUID,
    reporter_id: str = Form(..., max_length=64),
    answers: str = Form(..., description="Câu trả lời đã xác nhận hoặc sửa, JSON theo ReportAnswers"),
) -> dict[str, object]:
    """Section 16, steps D-H-I: the reporter confirms or corrects, and the report is re-scored.

    The plan's chart stops at "cập nhật kết quả". Corrected answers can change the
    route - for example to the verification queue - so the report is routed again.
    """
    database = require_database(request)
    report_answers = await with_server_reporter_level(database, parse_answers(answers), reporter_id)

    def location():
        with database.connect() as conn:
            return conn.execute("SELECT latitude, longitude FROM observations WHERE id = %s",
                                (observation_id,)).fetchone()

    where = await run_in_threadpool(location)
    nearby = None if where is None else await nearby_reports(
        database, where["latitude"], where["longitude"], report_answers, observation_id)

    def update():
        with database.connect() as conn:
            row = repository.get_observation(conn, observation_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Không có báo cáo này.")
            if row["reporter_id"] is None or row["reporter_id"] != reporter_id:
                raise HTTPException(status_code=403, detail="Chỉ người gửi báo cáo mới xác nhận được.")
            if row["status"] in ("verified", "dismissed"):
                raise HTTPException(status_code=409, detail="Báo cáo đã có kết luận xác minh cuối.")
            result = rescore(row["analysis"], report_answers, nearby)
            item = scored_observation(
                result,
                reporter_id=row["reporter_id"], latitude=row["latitude"], longitude=row["longitude"],
                location_accuracy_m=row["location_accuracy_m"], observed_at=row["observed_at"],
                image_path=row["image_path"], image_sha256=row["image_sha256"],
                image_phash_hex=f"{int(row['image_phash'], 2):016x}",
            )
            repository.update_scoring(conn, observation_id, item)
            task_id = verification.queue(conn, observation_id, result.review) if result.review.decision == "verify" else None
            if task_id is not None:
                verification.log(conn, task_id, reporter_id, "reporter", "answers_confirmed")
            return result, item, task_id

    result, item, task_id = await run_in_threadpool(update)
    return {
        "id": str(observation_id),
        "status": item.status,
        "verification_task_id": None if task_id is None else str(task_id),
        **result.payload,
    }


class VerdictIn(BaseModel):
    verdict: str = Field(description="confirmed | not_confirmed | unclear (unclear: tình nguyện viên)")
    note: str = Field("", max_length=2000)


def _task_call(database, action, *args):
    def run():
        with database.connect() as conn:
            return action(conn, *args)

    return run


async def _run_task(database, action, *args):
    try:
        return await run_in_threadpool(_task_call(database, action, *args))
    except verification.TaskError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def _then_refresh(action):
    """Run a verification step, then bring alerts in line in the same transaction."""

    def run(conn, *args):
        result = action(conn, *args)
        alerts.refresh(conn)
        return result

    return run


@app.get("/verification/tasks")
async def verification_tasks(
    request: Request,
    status: str | None = Query(None, pattern="^(open|claimed|escalated)$"),
    reviewer: Reviewer = Depends(current_reviewer),
) -> dict[str, object]:
    database = require_database(request)
    tasks = await _run_task(database, verification.list_tasks, reviewer, status)
    return {"reviewer": reviewer.__dict__, "tasks": tasks}


@app.post("/verification/tasks/{task_id}/claim")
async def claim_task(request: Request, task_id: UUID, reviewer: Reviewer = Depends(current_reviewer)):
    return await _run_task(require_database(request), verification.claim, task_id, reviewer)


@app.post("/verification/tasks/{task_id}/submit")
async def submit_task(request: Request, task_id: UUID, body: VerdictIn,
                      reviewer: Reviewer = Depends(current_reviewer)):
    return await _run_task(require_database(request), _then_refresh(verification.submit), task_id, reviewer,
                           body.verdict, body.note)


@app.post("/verification/tasks/{task_id}/resolve")
async def resolve_task(request: Request, task_id: UUID, body: VerdictIn,
                       reviewer: Reviewer = Depends(current_reviewer)):
    return await _run_task(require_database(request), _then_refresh(verification.resolve), task_id, reviewer,
                           body.verdict, body.note)


@app.post("/verification/tasks/{task_id}/photos", status_code=201)
async def task_photo(
    request: Request,
    task_id: UUID,
    image: UploadFile = File(...),
    kind: str = Form(..., pattern="^(before|after)$"),
    answers: str | None = Form(None, description="Tình trạng lúc chụp, ví dụ {\"water_stagnant\": \"no\"}"),
    reviewer: Reviewer = Depends(current_reviewer),
) -> dict[str, object]:
    """Before/after photos from the site (module flow: Before/After Verification -> Update Risk).

    The photo is scored with the report's own group rule, starting from the reporter's
    answers and overriding what the verifier saw on site, so 'after' can be compared
    with the original score on the dashboard.
    """
    database = require_database(request)
    require_detector(request)
    try:
        row = await run_in_threadpool(_task_call(database, verification.task_observation, task_id, reviewer))
    except verification.TaskError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    on_site = parse_answers(answers).model_dump(mode="json", exclude_defaults=True)
    merged = ReportAnswers.model_validate({**row["answers"], **on_site})
    picture = await read_photo(image)
    report = await quality_gate(picture)
    result = await run_pipeline(request.app.state, picture, merged, report,
                                conf_floor=0.05, with_water=True, with_vision=False)
    stored = await run_in_threadpool(save_photo, picture)
    group_score = result.vector_watch.group_score
    saved = await _run_task(database, verification.add_photo, task_id, reviewer, kind,
                            stored.relative_path, group_score, result.payload)
    return {**saved, "before_score": row["group_score"], "vector_watch": result.vector_watch.as_dict()}


@app.get("/vector-watch/catalog")
def vector_watch_groups() -> dict[str, object]:
    """Groups, tiers, the questions each group's form asks, and their limitations."""
    return vector_watch_catalog()


@app.get("/dashboard", include_in_schema=False)
def dashboard_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html", media_type="text/html; charset=utf-8")


@app.get("/dashboard/data")
async def dashboard_data(
    request: Request,
    hours: int = Query(24 * 14, ge=1, le=24 * 90),
    include_demo: bool = Query(False),
) -> dict[str, object]:
    database = require_database(request)
    domains = {"mosquito_breeding": "vector", "fly_organic": "sanitation", "water_anomaly": "water_quality"}

    def query():
        with database.connect() as conn:
            rows = repository.dashboard_points(conn, hours=hours, include_demo=include_demo)
            spots = {layer: repository.hotspots(conn, hours=min(hours, 72), category=domain,
                                                include_demo=include_demo)
                     for layer, domain in domains.items()}
        return rows, spots

    rows, spots = await run_in_threadpool(query)
    return {"hours": hours, **dashboard.build(rows, spots)}


@app.get("/verification/tasks/{task_id}/events")
async def task_events(request: Request, task_id: UUID, reviewer: Reviewer = Depends(current_reviewer)):
    return {"events": await _run_task(require_database(request), verification.events, task_id)}


@app.get("/observations")
async def list_observations(
    request: Request,
    hours: int = Query(72, ge=1, le=24 * 30),
    limit: int = Query(500, ge=1, le=5000),
    include_demo: bool = Query(False),
) -> dict[str, object]:
    database = require_database(request)

    def query():
        with database.connect() as conn:
            return repository.recent_observations(conn, hours=hours, limit=limit, include_demo=include_demo)

    return {"hours": hours, "observations": await run_in_threadpool(query)}


@app.get("/hotspots")
async def list_hotspots(
    request: Request,
    hours: int = Query(72, ge=1, le=24 * 30),
    radius_m: float = Query(500, gt=0, le=5000),
    min_reports: int = Query(3, ge=2, le=100),
    min_reporters: int = Query(2, ge=1, le=100),
    category: str = Query("vector", pattern="^(vector|environment|sanitation|water_quality|aquatic)$"),
    include_demo: bool = Query(False),
) -> dict[str, object]:
    """Section 13: at least `min_reports` reports within `radius_m` in the last `hours`."""
    database = require_database(request)
    params = {"hours": hours, "radius_m": radius_m, "min_reports": min_reports,
              "min_reporters": min_reporters, "category": category, "include_demo": include_demo}

    def query():
        with database.connect() as conn:
            found = repository.hotspots(conn, **params)
            counts = repository.cluster_counts(
                conn, **{k: v for k, v in params.items() if k != "min_reporters"})
        return found, counts

    found, counts = await run_in_threadpool(query)
    return {
        "parameters": params,
        "hotspots": found,
        "reports_considered": counts["reports"],
        # The plan's "cluster -1": reports with too few neighbours to form any cluster.
        "unclustered": counts["unclustered"],
    }


@app.get("/trends")
async def report_trend(
    request: Request,
    days: int = Query(7, ge=1, le=90),
    category: str = Query("vector", pattern="^(vector|environment)$"),
    latitude: float | None = Query(None, ge=-90, le=90),
    longitude: float | None = Query(None, ge=-180, le=180),
    radius_m: float = Query(1000, gt=0, le=20000),
    include_demo: bool = Query(False),
) -> dict[str, object]:
    """Section 15: this period against the one before, with a daily series for a chart.

    Give latitude and longitude to look at one area (a hotspot) instead of everywhere.
    """
    if (latitude is None) != (longitude is None):
        raise HTTPException(status_code=422, detail="Cần cả latitude và longitude, hoặc bỏ cả hai.")
    database = require_database(request)

    def query():
        with database.connect() as conn:
            return repository.daily_counts(
                conn, days=days, category=category, latitude=latitude, longitude=longitude,
                radius_m=radius_m, include_demo=include_demo)

    series = await run_in_threadpool(query)
    previous = sum(row["reports"] for row in series[:days])
    current = sum(row["reports"] for row in series[days:])
    return {
        "days": days,
        "category": category,
        "area": None if latitude is None else {"latitude": latitude, "longitude": longitude, "radius_m": radius_m},
        **classify(current, previous).as_dict(),
        "series": series,
        "note": "Số báo cáo phụ thuộc cả số người dùng app, không chỉ mức nguy cơ. "
                "Một đợt vận động báo cáo cũng làm số liệu tăng.",
    }




# --- Section 17: the rest of the API -------------------------------------------------

STATUS_TEXT = {
    "published": "Chưa xác minh",
    "needs_confirmation": "Chờ người gửi xác nhận",
    "needs_verification": "Đang chờ xác minh",
    "verified": "Đã xác minh",
    "dismissed": "Đã kiểm tra, không đúng",
    "pending": "Đang xử lý",
}
LEVEL_COLORS = {"green": "#2e7d32", "yellow": "#f9a825", "orange": "#ef6c00", "red": "#c62828"}
WRONG_ITEMS = ("detections", "water", "vision", "risk_level", "group", "location")


def group_label(group: str | None) -> str:
    return CATALOG[group].label if group in CATALOG else "Nước đọng, vật chứa"


def public_observation(row: dict) -> dict[str, object]:
    """One report as anyone may see it: no reporter id, no photo, coarse position."""
    analysis = row["analysis"] or {}
    watch = analysis.get("vector_watch") or {}
    return {
        "id": str(row["id"]),
        "observed_at": row["observed_at"],
        "latitude": round(row["latitude"], 3),
        "longitude": round(row["longitude"], 3),
        "type": group_label(row["vector_group"]),
        "vector_group": row["vector_group"],
        "level": row["level"],
        "risk": row["risk"],
        "confidence": row["confidence"],
        "status": row["status"],
        "status_label": STATUS_TEXT.get(row["status"], row["status"]),
        "group_score": row["group_score"],
        "group_score_after": row["group_score_after"],
        "statement": watch.get("statement"),
        "limitation": watch.get("limitation"),
        "review": analysis.get("review"),
        "detections": [{"label": d.get("label"), "confidence": d.get("confidence")}
                       for d in analysis.get("detections", [])],
        "disclaimer": DETECTION_DISCLAIMER,
    }


def _db_call(database, action):
    def run():
        with database.connect() as conn:
            return action(conn)

    return run


@app.get("/observations/{observation_id}")
async def observation_detail(request: Request, observation_id: UUID) -> dict[str, object]:
    database = require_database(request)
    row = await run_in_threadpool(_db_call(database, lambda conn: repository.get_observation(conn, observation_id)))
    if row is None:
        raise HTTPException(status_code=404, detail="Không có báo cáo này.")
    return public_observation(row)


@app.get("/observations/{observation_id}/photo", include_in_schema=False)
async def observation_photo(request: Request, observation_id: UUID,
                            reviewer: Reviewer = Depends(current_reviewer)) -> FileResponse:
    """Stored photos (faces already blurred) are for reviewers only, never public."""
    database = require_database(request)
    row = await run_in_threadpool(_db_call(database, lambda conn: repository.get_observation(conn, observation_id)))
    if row is None:
        raise HTTPException(status_code=404, detail="Không có báo cáo này.")
    path = photo_dir() / row["image_path"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Ảnh không còn trên máy chủ.")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/observations/{observation_id}/analyze")
async def reanalyze_observation(
    request: Request,
    observation_id: UUID,
    with_vision: bool = Query(False),
    reviewer: Reviewer = Depends(current_reviewer),
) -> dict[str, object]:
    """Run the models again on the stored photo (new model version, or add the vision model).

    Reviewers only: anyone else could use it to hammer the models. The answers stay the
    reporter's; the level and route are recomputed and saved.
    """
    database = require_database(request)
    require_detector(request)
    row = await run_in_threadpool(_db_call(database, lambda conn: repository.get_observation(conn, observation_id)))
    if row is None:
        raise HTTPException(status_code=404, detail="Không có báo cáo này.")
    if row["status"] in ("verified", "dismissed"):
        raise HTTPException(status_code=409, detail="Báo cáo đã có kết luận xác minh cuối.")
    path = photo_dir() / row["image_path"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Ảnh không còn trên máy chủ.")
    picture = await run_in_threadpool(load_image, path.read_bytes())
    report = await quality_gate(picture, skip=True)
    answers = ReportAnswers.model_validate(row["answers"])
    nearby = await nearby_reports(database, row["latitude"], row["longitude"], answers, observation_id)
    result = await run_pipeline(request.app.state, picture, answers, report, conf_floor=0.05,
                                with_water=True, with_vision=with_vision, nearby_reports=nearby)
    item = scored_observation(
        result, reporter_id=row["reporter_id"], latitude=row["latitude"], longitude=row["longitude"],
        location_accuracy_m=row["location_accuracy_m"], observed_at=row["observed_at"],
        image_path=row["image_path"], image_sha256=row["image_sha256"],
        image_phash_hex=f"{int(row['image_phash'], 2):016x}",
    )

    def update(conn):
        repository.update_scoring(conn, observation_id, item)
        task_id = verification.queue(conn, observation_id, result.review) if result.review.decision == "verify" else None
        if task_id is not None:
            verification.log(conn, task_id, reviewer.id, reviewer.role, "reanalyzed")
        alerts.refresh(conn)
        return task_id

    task_id = await run_in_threadpool(_db_call(database, update))
    return {"id": str(observation_id), "status": item.status,
            "verification_task_id": None if task_id is None else str(task_id), **result.payload}


class FeedbackIn(BaseModel):
    reporter_id: str | None = Field(None, max_length=64)
    wrong_items: list[str] = Field(default_factory=list, max_length=len(WRONG_ITEMS))
    note: str = Field("", max_length=1000)


@app.post("/observations/{observation_id}/feedback", status_code=201)
async def observation_feedback(request: Request, observation_id: UUID, body: FeedbackIn) -> dict[str, object]:
    """"Đánh dấu thông tin sai": kept as a label for retraining, and the report goes to review."""
    unknown = sorted(set(body.wrong_items) - set(WRONG_ITEMS))
    if unknown or not (body.wrong_items or body.note.strip()):
        raise HTTPException(status_code=422, detail=f"Chọn mục sai trong {WRONG_ITEMS} hoặc ghi chú.")
    database = require_database(request)

    def save(conn):
        row = repository.get_observation(conn, observation_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Không có báo cáo này.")
        conn.execute(
            "INSERT INTO observation_feedback (observation_id, reporter_id, wrong_items, note) VALUES (%s, %s, %s, %s)",
            (observation_id, body.reporter_id, body.wrong_items, body.note),
        )
        if row["status"] in ("verified", "dismissed"):
            return None
        route = ReviewRoute("verify", ["Người dùng báo kết quả AI sai: " + ", ".join(body.wrong_items or ["ghi chú"])],
                            expert_required=row["expert_required"])
        conn.execute("UPDATE observations SET status = 'needs_verification', needs_verification = true WHERE id = %s",
                     (observation_id,))
        return verification.queue(conn, observation_id, route)

    task_id = await run_in_threadpool(_db_call(database, save))
    return {"id": str(observation_id), "recorded": True,
            "verification_task_id": None if task_id is None else str(task_id)}


@app.get("/map/points")
async def map_points(
    request: Request,
    hours: int = Query(72, ge=1, le=24 * 30),
    include_demo: bool = Query(False),
) -> dict[str, object]:
    """GeoJSON for the public map; each point's popup fields are in its properties."""
    database = require_database(request)
    rows = await run_in_threadpool(_db_call(
        database, lambda conn: repository.map_points(conn, hours=hours, include_demo=include_demo)))
    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
        "properties": {
            "id": str(r["id"]),
            "type": group_label(r["vector_group"]),
            "observed_at": r["observed_at"],
            "nearby_reports": r["nearby_reports"],
            "level": r["level"],
            "color": LEVEL_COLORS[r["level"]],
            "risk": r["risk"],
            "confidence": r["confidence"],
            "status": r["status"],
            "status_label": ("Đã xử lý" if r["group_score_after"] is not None
                             else STATUS_TEXT.get(r["status"], r["status"])),
            "is_demo": r["is_demo"],
        },
    } for r in rows]
    return {"type": "FeatureCollection", "hours": hours, "precision_m": 110, "features": features}


@app.get("/alerts")
async def list_alerts(
    request: Request,
    status: str | None = Query(None, pattern="^(open|verified|resolved|dismissed)$"),
    include_demo: bool = Query(False),
) -> dict[str, object]:
    database = require_database(request)
    rows = await run_in_threadpool(_db_call(
        database, lambda conn: alerts.list_alerts(conn, status, include_demo)))
    for row in rows:
        row["label"] = alerts.CATEGORY_LABELS.get(row["category"], row["category"])
    return {"alerts": rows}


class AlertVerdictIn(BaseModel):
    verdict: str = Field(description="verified | resolved | dismissed")
    note: str = Field("", max_length=2000)


@app.post("/alerts/{alert_id}/verify")
async def verify_alert(request: Request, alert_id: UUID, body: AlertVerdictIn,
                       reviewer: Reviewer = Depends(current_reviewer)):
    return await _run_task(require_database(request), alerts.verify, alert_id, reviewer, body.verdict, body.note)


class TaskIn(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    description: str = Field("", max_length=2000)
    category: str = Field("vector", pattern="^(vector|sanitation|water_quality)$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_m: float = Field(500, gt=0, le=5000)
    target_count: int = Field(3, ge=1, le=50)
    reward_points: int = Field(100, ge=0, le=500)
    hours: int = Field(24, ge=1, le=24 * 14)
    alert_id: UUID | None = None


@app.post("/tasks", status_code=201)
async def create_task(request: Request, body: TaskIn, reviewer: Reviewer = Depends(current_reviewer)):
    deadline = datetime.now(UTC) + timedelta(hours=body.hours)
    fields = body.model_dump(exclude={"hours"})
    return await _run_task(require_database(request),
                           lambda conn: community.create_task(conn, reviewer, deadline=deadline, **fields))


@app.get("/tasks")
async def list_community_tasks(request: Request, include_demo: bool = Query(False)) -> dict[str, object]:
    database = require_database(request)
    return {"tasks": await run_in_threadpool(_db_call(
        database, lambda conn: community.list_tasks(conn, include_demo)))}


@app.post("/tasks/{task_id}/complete")
async def complete_community_task(request: Request, task_id: UUID, reviewer: Reviewer = Depends(current_reviewer)):
    return await _run_task(require_database(request), community.complete_task, task_id, reviewer)


@app.get("/dashboard/metrics")
async def dashboard_metrics(
    request: Request,
    days: int = Query(14, ge=1, le=90),
    include_demo: bool = Query(False),
) -> dict[str, object]:
    database = require_database(request)
    return await run_in_threadpool(_db_call(
        database, lambda conn: metrics.compute(conn, days=days, include_demo=include_demo)))


@app.get("/notifications")
async def list_notifications(
    request: Request,
    latitude: float | None = Query(None, ge=-90, le=90),
    longitude: float | None = Query(None, ge=-180, le=180),
    radius_m: float = Query(1000, gt=0, le=10000),
    reporter_id: str | None = Query(None, max_length=64),
    hours: int = Query(72, ge=1, le=24 * 30),
    include_demo: bool = Query(False),
) -> dict[str, object]:
    """Area notices around a point, and/or personal ones for a reporter id."""
    if (latitude is None) != (longitude is None):
        raise HTTPException(status_code=422, detail="Cần cả latitude và longitude, hoặc bỏ cả hai.")
    if latitude is None and reporter_id is None:
        raise HTTPException(status_code=422, detail="Cần vị trí hoặc reporter_id.")
    database = require_database(request)

    def query(conn):
        area = [] if latitude is None else notifications.for_area(
            conn, latitude, longitude, radius_m, hours, include_demo)
        personal = [] if reporter_id is None else notifications.for_user(conn, reporter_id, hours)
        return area, personal

    area, personal = await run_in_threadpool(_db_call(database, query))
    return {"area": area, "personal": personal}


@app.get("/community/profile/{user_id}")
async def community_profile(request: Request, user_id: str) -> dict[str, object]:
    if len(user_id) > 64:
        raise HTTPException(status_code=422, detail="Mã người dùng quá dài.")
    database = require_database(request)
    return await run_in_threadpool(_db_call(database, lambda conn: community.profile(conn, user_id)))


@app.get("/community/leaderboard")
async def community_leaderboard(
    request: Request, limit: int = Query(10, ge=1, le=50), include_demo: bool = Query(False),
) -> dict[str, object]:
    database = require_database(request)
    return {"leaders": await run_in_threadpool(_db_call(
        database, lambda conn: community.leaderboard(conn, limit, include_demo)))}



# --- Rivers and canals: segments, Risk Passport, Translator, FHIR (AquaPulse brief) ------

@app.get("/waterways")
async def waterway_overview(request: Request, include_demo: bool = Query(False)) -> dict[str, object]:
    """Every canal segment with its Eco / Risk / Confidence / Trend / Human-health indices and colour."""
    database = require_database(request)
    segments = await run_in_threadpool(_db_call(database, lambda conn: waterways.overview(conn, include_demo)))
    return {"segments": segments, "attribution": "Đường kênh © OpenStreetMap contributors (ODbL)"}


@app.get("/waterways/{segment_id}/passport")
async def waterway_passport(request: Request, segment_id: str, include_demo: bool = Query(False)) -> dict[str, object]:
    """The segment's Risk Passport: indices, explanation, history, anomalies, life, participation, rain, alerts."""
    database = require_database(request)
    result = await run_in_threadpool(_db_call(
        database, lambda conn: waterways.passport(conn, segment_id, include_demo)))
    if result is None:
        raise HTTPException(status_code=404, detail="Không có đoạn kênh này.")
    return result


@app.get("/translator")
async def health_translator(
    request: Request,
    question: str = Query(..., description="children | odor | after_rain | biodiversity"),
    segment_id: str | None = Query(None),
    include_demo: bool = Query(False),
) -> dict[str, object]:
    """Eco-Human Health Translator: a plain answer with its data, certainty and unknowns."""
    database = require_database(request)
    try:
        return await run_in_threadpool(_db_call(
            database, lambda conn: translator.answer(conn, question, segment_id, include_demo)))
    except translator.TranslatorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/translator/questions")
def translator_questions() -> dict[str, object]:
    return {"questions": translator.QUESTIONS, "needs_segment": list(translator.NEEDS_SEGMENT)}


FHIR_JSON = "application/fhir+json"


@app.get("/fhir/Observation/{observation_id}")
async def fhir_observation(request: Request, observation_id: UUID) -> JSONResponse:
    database = require_database(request)

    def query(conn):
        row = repository.get_observation(conn, observation_id)
        segment = None if row is None else waterways.segment_for(conn, row["latitude"], row["longitude"])
        return row, segment

    row, segment = await run_in_threadpool(_db_call(database, query))
    if row is None:
        raise HTTPException(status_code=404, detail="Không có báo cáo này.")
    return JSONResponse(jsonable_encoder(fhir.observation(row, segment["id"] if segment else None)),
                        media_type=FHIR_JSON)


@app.get("/fhir/waterways/{segment_id}")
async def fhir_segment_bundle(request: Request, segment_id: str, include_demo: bool = Query(False)) -> JSONResponse:
    """Bundle: the segment (Location), its passport (RiskAssessment) and its reports (Observation)."""
    database = require_database(request)

    def query(conn):
        p = waterways.passport(conn, segment_id, include_demo, with_rain=False)
        if p is None:
            return None, []
        ids = [r["id"] for r in waterways._reports(conn, 30, include_demo, segment_id)]
        rows = conn.execute("SELECT * FROM observations WHERE id = ANY(%s) ORDER BY observed_at", (ids,)).fetchall()
        return p, rows

    p, rows = await run_in_threadpool(_db_call(database, query))
    if p is None:
        raise HTTPException(status_code=404, detail="Không có đoạn kênh này.")
    resources = [fhir.location_for_segment(p["segment"]),
                 fhir.risk_assessment(p, [str(r["id"]) for r in rows]),
                 *[fhir.observation(r, segment_id) for r in rows]]
    return JSONResponse(jsonable_encoder(fhir.bundle(resources)), media_type=FHIR_JSON)


# --- OneAquaHealth: the consortium's own sites and indicators ----------------------------

@app.get("/oneaqua/cities")
async def oneaqua_cities(request: Request) -> dict[str, object]:
    """The project's pilot cities, with how many sites and their average health-risk score."""
    database = require_database(request)
    rows = await run_in_threadpool(_db_call(database, oneaqua.cities))
    return {"cities": rows, "attribution": oneaqua.ATTRIBUTION, "source": oneaqua.base_url()}


@app.get("/oneaqua/sites")
async def oneaqua_sites(
    request: Request,
    city: str | None = Query(None, max_length=8),
    kind: str | None = Query(None, pattern="^(research|citizen)$"),
) -> dict[str, object]:
    """Their monitoring sites as GeoJSON: research sites, and sites made in their Citizen Science App."""
    database = require_database(request)
    rows = await run_in_threadpool(_db_call(database, lambda conn: oneaqua.sites(conn, city, kind)))
    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [r["longitude"], r["latitude"]]},
        "properties": {**{k: v for k, v in r.items() if k not in ("latitude", "longitude")},
                       "health_risk_word": oneaqua.risk_word(r["health_risk"])},
    } for r in rows]
    return {"type": "FeatureCollection", "features": features, "attribution": oneaqua.ATTRIBUTION,
            "note": "Số liệu do OneAquaHealth đo và công bố. AquaPulse chỉ hiển thị, không tính lại, "
                    "không trộn vào điểm rủi ro từ quan sát của người dân."}


@app.get("/oneaqua/sites/{code}")
async def oneaqua_site(request: Request, code: str) -> dict[str, object]:
    """One site: their indicators explained in plain words, their sampling history, nitrates and weather."""
    database = require_database(request)

    def query(conn):
        return oneaqua.site(conn, code), oneaqua.history(conn, code)

    row, history = await run_in_threadpool(_db_call(database, query))
    if row is None:
        raise HTTPException(status_code=404, detail="Không có điểm quan trắc này.")
    weather = await run_in_threadpool(oneaqua.weather, code)
    nitrates = await run_in_threadpool(oneaqua.nitrates, code)
    return {"site": row, "explanation": oneaqua.explain(row), "history": history,
            "nitrates": nitrates, "weather": weather, "labels": oneaqua.INDICATOR_LABELS,
            "attribution": oneaqua.ATTRIBUTION}


@app.post("/oneaqua/sync")
async def oneaqua_sync(request: Request, reviewer: Reviewer = Depends(current_reviewer)) -> dict[str, object]:
    """Pull their API now (experts only; the server also does it on a timer)."""
    if reviewer.role != "expert":
        raise HTTPException(status_code=403, detail="Chỉ chuyên gia mới chạy đồng bộ dữ liệu OneAquaHealth.")
    database = require_database(request)
    try:
        return await run_in_threadpool(oneaqua.sync, database)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=502, detail=f"Không lấy được dữ liệu OneAquaHealth: {exc}") from exc


# --- What the press reported about the canals -------------------------------------------

@app.get("/press")
async def press_mentions(
    request: Request,
    waterway: str | None = Query(None, max_length=80),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, object]:
    """Articles about Ho Chi Minh City's canals since the start of the year, by waterway.
    Context only: a headline never enters a score, a hotspot or an alert."""
    database = require_database(request)

    def query(conn):
        return press.mentions(conn, waterway, limit), press.summary(conn)

    rows, overview = await run_in_threadpool(_db_call(database, query))
    return {"mentions": rows, **overview}


@app.post("/press/sync")
async def press_sync(request: Request, reviewer: Reviewer = Depends(current_reviewer)) -> dict[str, object]:
    if reviewer.role != "expert":
        raise HTTPException(status_code=403, detail="Chỉ chuyên gia mới chạy đồng bộ tin báo chí.")
    database = require_database(request)
    try:
        return await run_in_threadpool(press.sync, database)
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=502, detail=f"Không lấy được tin báo chí: {exc}") from exc
