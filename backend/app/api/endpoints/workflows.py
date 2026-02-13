from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from app.models.workflow import Workflow, WorkflowCreate, WorkflowRead
from app.core.config import settings
from sqlmodel import create_engine

# Database setup (simplified for prototype)
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})

def get_session():
    with Session(engine) as session:
        yield session

router = APIRouter()

@router.post("/", response_model=WorkflowRead)
def create_workflow(workflow: WorkflowCreate, session: Session = Depends(get_session)):
    db_workflow = Workflow.from_orm(workflow)
    session.add(db_workflow)
    session.commit()
    session.refresh(db_workflow)
    return db_workflow

@router.get("/", response_model=List[WorkflowRead])
def read_workflows(offset: int = 0, limit: int = 100, session: Session = Depends(get_session)):
    workflows = session.exec(select(Workflow).offset(offset).limit(limit)).all()
    return workflows

@router.get("/{workflow_id}", response_model=WorkflowRead)
def read_workflow(workflow_id: int, session: Session = Depends(get_session)):
    workflow = session.get(Workflow, workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow
