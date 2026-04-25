from flask import Flask, send_from_directory

from backend.api.routes import api_bp
from backend.config import Config, ensure_directories


def create_app() -> Flask:
    ensure_directories()

    app = Flask(
        __name__,
        static_folder=str(Config.FRONTEND_DIR),
        static_url_path="",
    )
    app.config.from_object(Config)
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/")
    def index():
        return send_from_directory(str(Config.FRONTEND_DIR), "index.html")

    @app.get("/assets/<path:filename>")
    def assets(filename: str):
        return send_from_directory(str(Config.OUTPUT_DIR), filename)

    return app


app = create_app()
