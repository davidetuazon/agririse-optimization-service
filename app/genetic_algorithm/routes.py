from fastapi import APIRouter, Body, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict
from fastapi.responses import JSONResponse

from app.genetic_algorithm.executor import acquire
from app.genetic_algorithm.core.ga_engine import run_ga

router = APIRouter()

class Coverage(BaseModel):
    barangay: str
    fractionalAreaHa: float

class CanalInput(BaseModel):
    _id: str
    mainLateralId: str
    tbsByDamHa: float
    netWaterDemandM3: float
    seepageM3: float
    lossFactorPercentage: float
    coverage: List[Coverage]

class GAInput(BaseModel):
    runId: str
    scenario: str
    cropVariant: str
    totalSeasonalWaterSupplyM3: float
    readings: Dict[str, float]
    canalInput: List[CanalInput]


@router.post('/ga')

async def optimize_ga(payload: GAInput = Body(...), background_tasks: BackgroundTasks = None):
    # destructure payload
    scenario: str = payload.scenario
    crop_variant: str = payload.cropVariant
    readings: list = payload.readings

    canal_input_list = [c.model_dump() for c in payload.canalInput]
    total_water_available = float(payload.totalSeasonalWaterSupplyM3)

    if not acquire():
        return JSONResponse(
            status_code=503,
            content={
                'status': 'failed',
                'runId': payload.runId,
                'reason': 'All workers are busy. Retry shortly.'
            }
        )
    # try:
    #     await acquire()
    # except asyncio.TimeoutError:
    #     return JSONResponse(
    #         status_code=503,
    #         content={
    #             'status': 'failed',
    #             'runId': payload.runId,
    #             'reason': 'All workers are busy. Retry shortly.'
    #         }
    #     )
    
    background_tasks.add_task(
        run_ga,
        canal_input_list,
        total_water_available,
        payload.runId,
    )

    return { 'status': 'pending', 'runId': payload.runId }