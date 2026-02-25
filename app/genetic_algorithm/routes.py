from fastapi import APIRouter, Body, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import asyncio

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

    canal_input: list = payload.canalInput
    canal_input_list: list = [c.model_dump() for c in canal_input]
    total_water_available: float = float(payload.totalSeasonalWaterSupplyM3)

    background_tasks.add_task(
        run_ga,
        canal_input_list,
        total_water_available,
        payload.runId,
    )

    return { 'status': 'pending', 'runId': payload.runId }