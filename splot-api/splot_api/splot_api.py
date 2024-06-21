"""Main module."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routers import (
    sparams,
    files,
)

ORIGINS = ["*"]


def initialize() -> FastAPI:
    # Create the api, add CORS origins, and set up the routes
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(sparams.router)
    app.include_router(files.router)
    return app


app = initialize()

if __name__ == "__main__":
    uvicorn.run("splot_api:app", port=8080, host="0.0.0.0", reload=True)
