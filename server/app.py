"""
OpenEnv-compliant server entry point.

This shim re-exports the FastAPI ``app`` from the main package so that
``openenv validate`` finds ``server/app.py`` at the project root *and* a
callable ``main()`` function — both are hard requirements of the multi-mode
deployment validator.

Usage:
    uv run server                         # via [project.scripts]
    uvicorn server.app:app --reload       # development
    python server/app.py                  # direct
"""

from ticketflow_env.server.app import app  # noqa: F401


def main() -> None:
    """Entry point used by ``[project.scripts] server``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
