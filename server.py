"""
Lightweight HTTP server for the Sentiment Analysis Web UI.
Serves static files + handles /analyze POST endpoint.
"""

import os
import json
import tempfile
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from sentiment_analyzer import analyze_csv_detailed

PORT = 8080
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class SentimentHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")

    # ── Static file serving ─────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_file(os.path.join(STATIC_DIR, "index.html"), "text/html")
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            full = os.path.join(STATIC_DIR, rel)
            mt, _ = mimetypes.guess_type(full)
            self._serve_file(full, mt or "application/octet-stream")
        else:
            self._serve_file(os.path.join(STATIC_DIR, "index.html"), "text/html")

    def _serve_file(self, filepath, content_type):
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self._json_error(404, "File not found")

    # ── /analyze POST endpoint ──────────────────────────────────
    def do_POST(self):
        if self.path == "/analyze":
            self._handle_analyze()
        else:
            self._json_error(404, "Unknown endpoint")

    def _handle_analyze(self):
        try:
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Parse multipart form data
            if "multipart/form-data" in content_type:
                boundary_match = content_type.split("boundary=")
                if len(boundary_match) < 2:
                    self._json_error(400, "Missing boundary")
                    return
                boundary = boundary_match[1].strip().encode()
                fields, files = self._parse_multipart(body, boundary)
            else:
                self._json_error(415, "Expected multipart/form-data")
                return

            # Get text column name (optional)
            text_col = fields.get("text_column", ["text"])[0]

            # Get uploaded file
            if "csv_file" not in files:
                self._json_error(400, "No csv_file field found")
                return

            file_data = files["csv_file"]["data"]

            # Save to temp file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".csv", mode="wb"
            ) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name

            try:
                result = analyze_csv_detailed(tmp_path, text_col)
                # DEBUG LOGGING
                with open("server_debug.log", "w", encoding="utf-8") as log_f:
                    log_f.write(json.dumps(result, indent=2))
            finally:
                os.unlink(tmp_path)

            self._json_ok(result)

        except Exception as e:
            self._json_error(500, str(e))

    def _parse_multipart(self, body: bytes, boundary: bytes):
        """Parse multipart/form-data body manually."""
        fields = {}
        files = {}
        delimiter = b"--" + boundary
        end_delimiter = delimiter + b"--"

        parts = body.split(delimiter)
        for part in parts[1:]:  # skip preamble
            if part.strip() == b"--" or part.strip() == b"":
                continue
            if part.startswith(b"--"):
                break

            # Split headers from body
            if b"\r\n\r\n" in part:
                header_section, content = part.split(b"\r\n\r\n", 1)
            elif b"\n\n" in part:
                header_section, content = part.split(b"\n\n", 1)
            else:
                continue

            # Remove trailing CRLF
            if content.endswith(b"\r\n"):
                content = content[:-2]
            elif content.endswith(b"\n"):
                content = content[:-1]

            headers = {}
            for line in header_section.split(b"\r\n"):
                if b":" in line:
                    k, v = line.split(b":", 1)
                    headers[k.strip().lower().decode()] = v.strip().decode()

            disposition = headers.get("content-disposition", "")
            name = None
            filename = None
            for segment in disposition.split(";"):
                segment = segment.strip()
                if segment.startswith("name="):
                    name = segment[5:].strip('"')
                elif segment.startswith("filename="):
                    filename = segment[9:].strip('"')

            if name is None:
                continue

            if filename:
                files[name] = {"filename": filename, "data": content}
            else:
                fields.setdefault(name, []).append(content.decode("utf-8", errors="replace"))

        return fields, files

    def _json_ok(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, code, message):
        body = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


if __name__ == "__main__":
    os.makedirs(STATIC_DIR, exist_ok=True)
    httpd = HTTPServer(("localhost", PORT), SentimentHandler)
    print(f"[OK] Sentiment Analysis Server running at http://localhost:{PORT}")
    print("   Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
