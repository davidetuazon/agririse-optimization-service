from fastapi import FastAPI
from app.genetic_algorithm.routes import router as optimization_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Irrigation Optimization Service",
        version="1.0.0",
        description="Genetic Algorithm–based irrigation water allocation optimizer"
    )

    app.include_router(
        optimization_router,
        prefix="/api/v1/optimization",
        tags=["Optimization"]
    )

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app

app = create_app()
