from typing import Optional, List, Dict
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON

class WorkflowBase(SQLModel):
    title: str
    description: Optional[str] = None
    steps_definition: List[Dict] = Field(default=[], sa_column=Column(JSON))

class Workflow(WorkflowBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    executions: List["Execution"] = Relationship(back_populates="workflow")

class WorkflowCreate(WorkflowBase):
    pass

class WorkflowRead(WorkflowBase):
    id: int
    created_at: datetime
    updated_at: datetime

class ExecutionBase(SQLModel):
    workflow_id: int = Field(foreign_key="workflow.id")
    status: str = "pending"  # pending, running, completed, failed
    logs: List[Dict] = Field(default=[], sa_column=Column(JSON))
    result: Optional[Dict] = Field(default={}, sa_column=Column(JSON))

class Execution(ExecutionBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    workflow: Optional[Workflow] = Relationship(back_populates="executions")

class ExecutionCreate(ExecutionBase):
    pass

class ExecutionRead(ExecutionBase):
    id: int
    started_at: datetime
    completed_at: Optional[datetime]
