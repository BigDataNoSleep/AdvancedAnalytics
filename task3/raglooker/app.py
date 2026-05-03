from typing import Any

from flask import Flask, Response, jsonify, render_template, request
from recommender import create_search_engine


def create_app() -> Flask:
    app = Flask(__name__)
    search_engine = create_search_engine()

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/api/search")
    def search() -> Any:
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            return jsonify({"error": "A game description is required."}), 400
        try:
            return Response(
                search_engine.search_stream(query),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        except Exception as exc:
            return jsonify({"error": "Search failed.", "details": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
