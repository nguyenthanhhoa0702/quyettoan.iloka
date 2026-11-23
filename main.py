import os
import re
import io
import csv
import json
import base64
import unicodedata
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv, set_key

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

ENV_PATH = ".env"
load_dotenv(ENV_PATH)

app = FastAPI(title="Bill OCR & Google Sheet Sync - Auto Map")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SERVICE_ACCOUNT_PATH = "service_account.json"
TOKEN_PATH = "token.json"
CREDS_PATH = "credentials.json"
LOG_FILE = "audit_logs.jsonl"
DEFAULT_TAB_NAME = "Cửa hàng"

runtime_config = {
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
}


# ================= MODELS =================
class SettingsPayload(BaseModel):
    api_key: str
    model: str


class OcrPayload(BaseModel):
    image_base64: str
    mime_type: str = "image/png"
    sheet_url: Optional[str] = ""
    sheet_name: Optional[str] = DEFAULT_TAB_NAME
    header_row: int = 3


class SheetCheckPayload(BaseModel):
    sheet_url: str


class PreviewPayload(BaseModel):
    sheet_url: str
    sheet_name: str = DEFAULT_TAB_NAME
    header_row: int = 3
    store_id: str
    ngay_tt: str
    tien_mat: Optional[float] = 0
    chuyen_khoan: Optional[float] = 0
    quang_cao: Optional[float] = 0
    ma_gd: Optional[str] = ""
    noi_dung: Optional[str] = ""


class ApprovePayload(PreviewPayload):
    target_row: int
    target_start_col: int
    cell_range_a1: str
    store_name: str
    batch_number: int


# ================= HELPERS =================
def normalize_vn(text: str) -> str:
    """Chuẩn hóa Unicode NFC, viết thường và bỏ khoảng trắng thừa"""
    return unicodedata.normalize("NFC", text).strip().lower()


def extract_spreadsheet_id(url: str) -> str:
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        raise HTTPException(status_code=400, detail="Link Google Sheet không hợp lệ!")
    return match.group(1)


def col_to_letter(col_idx: int) -> str:
    letter = ""
    while col_idx > 0:
        col_idx, rem = divmod(col_idx - 1, 26)
        letter = chr(65 + rem) + letter
    return letter


def get_google_creds():
    # Ưu tiên 1: Dùng Service Account (Không cần login, không bao giờ hết hạn token)
    if os.path.exists(SERVICE_ACCOUNT_PATH):
        return service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_PATH, scopes=SCOPES
        )

    # Ưu tiên 2: Fallback về OAuth token cá nhân nếu chưa có service_account.json
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDS_PATH):
                raise HTTPException(
                    status_code=400,
                    detail="Thiếu file service_account.json hoặc credentials.json trong thư mục project!",
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())
    return creds


def resolve_store_tab(service, spreadsheet_id: str) -> str:
    """Tự động dò tìm và khóa cứng vào tab 'Cửa hàng' (chuẩn hóa Unicode tiếng Việt)"""
    meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]
    target_norm = normalize_vn("Cửa hàng")

    for t in tabs:
        if target_norm in normalize_vn(t):
            return t

    raise HTTPException(
        status_code=400,
        detail=f"Không tìm thấy tab 'Cửa hàng' trong file! Các tab hiện có: {tabs}",
    )


def append_local_log(entry: dict):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def fetch_store_directory(sheet_url: str, header_row: int) -> list[dict]:
    """Lấy danh sách {id, name} từ tab Cửa hàng để AI tự động map"""
    if not sheet_url:
        return []
    try:
        spreadsheet_id = extract_spreadsheet_id(sheet_url)
        creds = get_google_creds()
        service = build("sheets", "v4", credentials=creds)
        target_tab = resolve_store_tab(service, spreadsheet_id)

        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f"'{target_tab}'!A:E",
        ).execute()
        rows = result.get("values", [])
        if len(rows) < header_row:
            return []

        headers = [normalize_vn(str(h)) for h in rows[header_row - 1]]
        id_col = 1
        for idx, h in enumerate(headers):
            if h == "id":
                id_col = idx
                break

        stores = []
        for r in rows[header_row:]:
            if len(r) > id_col:
                sid = re.sub(r"\s+", "", str(r[id_col]))
                sname = str(r[id_col + 1]).strip() if len(r) > (id_col + 1) else ""
                if sid:
                    stores.append({"id": sid, "name": sname})
        return stores
    except Exception:
        return []


# ================= ROUTES =================
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/config")
def get_config():
    is_connected = os.path.exists(SERVICE_ACCOUNT_PATH) or os.path.exists(TOKEN_PATH)
    return {
        "api_key": runtime_config["api_key"],
        "model": runtime_config["model"],
        "google_logged_in": is_connected,
        "auth_mode": "service_account" if os.path.exists(SERVICE_ACCOUNT_PATH) else "oauth",
    }


@app.post("/api/config")
def update_config(payload: SettingsPayload):
    runtime_config["api_key"] = payload.api_key.strip()
    runtime_config["model"] = payload.model.strip()

    if not os.path.exists(ENV_PATH):
        open(ENV_PATH, "w").close()
    set_key(ENV_PATH, "GEMINI_API_KEY", runtime_config["api_key"])
    set_key(ENV_PATH, "GEMINI_MODEL", runtime_config["model"])

    return {"status": "ok", "message": f"Đã cập nhật Key & Model ({runtime_config['model']}) không cần restart!"}


@app.post("/api/google/login")
def trigger_google_login(force_relogin: bool = False):
    if os.path.exists(SERVICE_ACCOUNT_PATH):
        return {
            "status": "ok",
            "message": "Hệ thống đang dùng Service Account (Tự động kết nối, không cần đăng nhập thủ công)!",
        }
    if force_relogin and os.path.exists(TOKEN_PATH):
        os.remove(TOKEN_PATH)
    get_google_creds()
    return {"status": "ok", "message": "Đăng nhập tài khoản Google thành công!"}


@app.post("/api/sheets/verify")
def verify_sheet_tabs(payload: SheetCheckPayload):
    spreadsheet_id = extract_spreadsheet_id(payload.sheet_url)
    creds = get_google_creds()
    service = build("sheets", "v4", credentials=creds)

    try:
        meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Không truy cập được file Sheet (Nhớ Share quyền Editor cho email Service Account): {str(e)}",
        )

    title = meta.get("properties", {}).get("title", "Google Sheet")
    tabs = [s["properties"]["title"] for s in meta.get("sheets", [])]
    target_tab = resolve_store_tab(service, spreadsheet_id)

    return {
        "title": title,
        "tabs": tabs,
        "target_tab": target_tab,
        "has_store_tab": True,
    }


@app.post("/api/ocr")
def run_ocr(payload: OcrPayload):
    if not runtime_config["api_key"]:
        raise HTTPException(status_code=400, detail="Chưa cấu hình GEMINI_API_KEY trong phần Cài đặt!")

    try:
        client = genai.Client(api_key=runtime_config["api_key"])
        raw_b64 = payload.image_base64.strip()
        mime_type = (payload.mime_type or "").strip()

        if raw_b64.startswith("data:"):
            header, raw_b64 = raw_b64.split(",", 1)
            extracted_mime = header.split(";")[0].replace("data:", "").strip()
            if extracted_mime:
                mime_type = extracted_mime

        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/png"

        image_bytes = base64.b64decode(raw_b64)

        store_list = fetch_store_directory(payload.sheet_url, payload.header_row)
        store_context = ""
        if store_list:
            lines = [f"- ID: {s['id']} | Tên: {s['name']}" for s in store_list[:500]]
            store_context = "\nDANH SÁCH CỬA HÀNG ĐỂ ĐỐI CHIẾU:\n" + "\n".join(lines)

        prompt = f"""Trích xuất thông tin từ ảnh bill/phiếu thanh toán này và tự động xác định ID Cửa hàng.
        {store_context}

        Quy tắc xác định "store_id":
        1. Tìm mã số ID cửa hàng xuất hiện trong nội dung chuyển khoản / ghi chú trên bill (ví dụ: '2241', 'CH2241', 'ID 2241' -> lấy '2241').
        2. Nếu nội dung không ghi số ID nhưng có tên cửa hàng hoặc tên người gửi khớp với một cửa hàng trong DANH SÁCH CỬA HÀNG ĐỂ ĐỐI CHIẾU ở trên, hãy trả về đúng mã ID của cửa hàng đó.
        3. Nếu hoàn toàn không xác định được, để chuỗi rỗng "".

        Trả về DUY NHẤT một JSON hợp lệ gồm các trường:
        - "store_id": chuỗi ID cửa hàng tìm được (hoặc "")
        - "ngay_tt": chuỗi ngày giao dịch định dạng DD/MM/YYYY
        - "so_tien": số nguyên (chỉ lấy chữ số, bỏ dấu chấm/phẩy/VNĐ)
        - "hinh_thuc": "CK" nếu là chuyển khoản ngân hàng/ví điện tử, "TM" nếu là tiền mặt
        - "ma_gd": chuỗi mã giao dịch / mã tham chiếu (nếu có)
        - "noi_dung": nội dung chuyển khoản hoặc ghi chú trên bill
        """

        response = client.models.generate_content(
            model=runtime_config["model"],
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi OCR ({runtime_config['model']}): {str(e)}")


@app.post("/api/preview")
def preview_change(payload: PreviewPayload):
    spreadsheet_id = extract_spreadsheet_id(payload.sheet_url)
    store_id = re.sub(r"\s+", "", str(payload.store_id))
    if not store_id:
        raise HTTPException(status_code=400, detail="Vui lòng nhập ID Cửa hàng!")

    ngay_str = payload.ngay_tt.strip()
    month_match = re.match(r"^\d{1,2}/(\d{1,2})/\d{2,4}$", ngay_str)
    if not month_match:
        raise HTTPException(
            status_code=400,
            detail=f"Ngày TT '{ngay_str}' chưa đúng định dạng DD/MM/YYYY (VD: 23/11/2026) để xác định cột Tháng!",
        )
    target_month = int(month_match.group(1))
    if not (1 <= target_month <= 12):
        raise HTTPException(status_code=400, detail=f"Tháng '{target_month}' không hợp lệ (phải từ 1 đến 12)!")

    creds = get_google_creds()
    service = build("sheets", "v4", credentials=creds)
    target_tab = resolve_store_tab(service, spreadsheet_id)

    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f"'{target_tab}'!A:ZZ",
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được tab '{target_tab}': {str(e)}")

    rows = result.get("values", [])
    if len(rows) < payload.header_row:
        raise HTTPException(
            status_code=400,
            detail=f"Không tìm thấy dòng tiêu đề ở dòng {payload.header_row} trên tab '{target_tab}'.",
        )

    headers = [normalize_vn(str(h)) for h in rows[payload.header_row - 1]]

    id_col_idx = 1
    for idx, h in enumerate(headers):
        if h == "id":
            id_col_idx = idx
            break

    first_month_col_idx = id_col_idx + 3
    for idx in range(id_col_idx + 1, len(headers)):
        if normalize_vn("ngày") in headers[idx]:
            first_month_col_idx = idx
            break

    target_row_idx = -1
    row_data = []
    for idx in range(payload.header_row, len(rows)):
        r = rows[idx]
        if len(r) > id_col_idx:
            cell_val = re.sub(r"\s+", "", str(r[id_col_idx]))
            if cell_val.lower() == store_id.lower():
                target_row_idx = idx + 1
                row_data = r
                break

    if target_row_idx == -1:
        col_letter = col_to_letter(id_col_idx + 1)
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy ID '{store_id}' trong cột {col_letter} của tab '{target_tab}'!",
        )

    store_name = row_data[id_col_idx + 1] if len(row_data) > (id_col_idx + 1) else "(Chưa đặt tên)"
    status_text = row_data[id_col_idx + 2] if len(row_data) > (id_col_idx + 2) else "(Trống)"

    month_col_idx = first_month_col_idx + (target_month - 1) * 4
    target_start_col = month_col_idx + 1

    c_ngay = row_data[month_col_idx].strip() if month_col_idx < len(row_data) else ""
    c_tm = row_data[month_col_idx + 1].strip() if (month_col_idx + 1) < len(row_data) else ""
    c_ck = row_data[month_col_idx + 2].strip() if (month_col_idx + 2) < len(row_data) else ""
    c_qc = row_data[month_col_idx + 3].strip() if (month_col_idx + 3) < len(row_data) else ""

    warnings = []
    if any([c_ngay, c_tm, c_ck, c_qc]):
        warnings.append(
            f"Tháng {target_month} của cửa hàng này ĐÃ CÓ DỮ LIỆU (Ngày: {c_ngay or '-'}, TM: {c_tm or 0}, CK: {c_ck or 0}, QC: {c_qc or 0}). Bấm Approve sẽ ghi đè lên!"
        )

    if payload.ma_gd and os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if f'"ma_gd": "{payload.ma_gd.strip()}"' in line:
                    warnings.append(f"Mã giao dịch '{payload.ma_gd}' đã từng được duyệt trước đó!")
                    break

    start_col_letter = col_to_letter(target_start_col)
    end_col_letter = col_to_letter(target_start_col + 3)
    cell_range_a1 = f"{start_col_letter}{target_row_idx}:{end_col_letter}{target_row_idx}"

    return {
        "sheet_name": target_tab,
        "target_row": target_row_idx,
        "target_start_col": target_start_col,
        "cell_range_a1": cell_range_a1,
        "store_name": store_name,
        "current_status": status_text,
        "batch_number": target_month,
        "warning": " | ".join(warnings),
    }


@app.post("/api/approve")
def approve_change(payload: ApprovePayload):
    spreadsheet_id = extract_spreadsheet_id(payload.sheet_url)
    creds = get_google_creds()
    service = build("sheets", "v4", credentials=creds)
    target_tab = resolve_store_tab(service, spreadsheet_id)

    values = [[
        payload.ngay_tt,
        payload.tien_mat if payload.tien_mat else "",
        payload.chuyen_khoan if payload.chuyen_khoan else "",
        payload.quang_cao if payload.quang_cao else "",
    ]]

    range_to_update = f"'{target_tab}'!{payload.cell_range_a1}"

    try:
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_to_update,
            valueInputOption="USER_ENTERED",
            body={"values": values},
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi ghi vào tab '{target_tab}': {str(e)}")

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tab": target_tab,
        "store_id": payload.store_id,
        "store_name": payload.store_name,
        "batch": f"Tháng {payload.batch_number}",
        "range": payload.cell_range_a1,
        "ngay_tt": payload.ngay_tt,
        "tien_mat": payload.tien_mat or 0,
        "chuyen_khoan": payload.chuyen_khoan or 0,
        "quang_cao": payload.quang_cao or 0,
        "ma_gd": payload.ma_gd or "",
        "noi_dung": payload.noi_dung or "",
    }
    append_local_log(log_entry)

    return {
        "status": "ok",
        "message": f"Đã ghi thành công vào tab [{target_tab}] ô {payload.cell_range_a1} ({payload.store_name})",
        "log": log_entry,
    }


@app.get("/api/logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return list(reversed(lines[-50:]))


@app.get("/api/logs/export")
def export_logs_csv():
    """Xuất toàn bộ lịch sử Audit Logs ra file CSV chuẩn UTF-8 BOM cho Excel"""
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="Chưa có dữ liệu log nào để xuất!")

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise HTTPException(status_code=404, detail="File log đang trống!")

    output = io.StringIO()
    output.write("\ufeff")

    writer = csv.writer(output)
    writer.writerow([
        "Thời gian duyệt",
        "Tab Sheet",
        "ID Cửa Hàng",
        "Tên Cửa Hàng",
        "Đợt / Tháng",
        "Vùng ô ghi",
        "Ngày TT",
        "Tiền mặt",
        "Chuyển khoản",
        "Quảng cáo",
        "Mã GD",
        "Nội dung CK",
    ])

    for r in records:
        writer.writerow([
            r.get("timestamp", ""),
            r.get("tab", "Cửa hàng"),
            r.get("store_id", ""),
            r.get("store_name", ""),
            r.get("batch", ""),
            r.get("range", ""),
            r.get("ngay_tt", ""),
            r.get("tien_mat", 0),
            r.get("chuyen_khoan", 0),
            r.get("quang_cao", 0),
            r.get("ma_gd", ""),
            r.get("noi_dung", ""),
        ])

    output.seek(0)
    filename = f"audit_logs_cuahang_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )