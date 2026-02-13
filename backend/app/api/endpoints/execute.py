from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session
from app.models.workflow import Execution, ExecutionCreate, ExecutionRead, Workflow
from app.services.workflow_engine import workflow_engine
from app.api.endpoints.workflows import get_session

router = APIRouter()

async def run_workflow_background(execution_id: int, workflow_id: int):
    # Re-create session for background task
    from app.api.endpoints.workflows import engine
    with Session(engine) as session:
        execution = session.get(Execution, execution_id)
        workflow = session.get(Workflow, workflow_id)
        if execution and workflow:
            updated_execution = await workflow_engine.execute_workflow(execution, workflow)
            session.add(updated_execution)
            session.commit()

@router.post("/", response_model=ExecutionRead)
def execute_workflow_endpoint(
    execution_request: ExecutionCreate, 
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    workflow = session.get(Workflow, execution_request.workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    db_execution = Execution.from_orm(execution_request)
    session.add(db_execution)
    session.commit()
    session.refresh(db_execution)
    
    background_tasks.add_task(run_workflow_background, db_execution.id, workflow.id)
    
    return db_execution

@router.get("/{execution_id}", response_model=ExecutionRead)
def get_execution_status(execution_id: int, session: Session = Depends(get_session)):
    execution = session.get(Execution, execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    return execution
