from datetime import datetime
from app.models.workflow import Workflow, Execution
from app.services.llm_service import llm_service

class WorkflowEngine:
    async def execute_workflow(self, execution: Execution, workflow: Workflow):
        execution.status = "running"
        execution.started_at = datetime.utcnow()
        execution.logs = []
        execution.result = {}
        
        context_data = {}
        
        try:
            for step in workflow.steps_definition:
                step_id = step.get("id")
                step_type = step.get("type")
                step_prompt = step.get("prompt")
                
                # Replace placeholders in prompt with context data if any
                # Simple placeholder replacement {key}
                formatted_prompt = step_prompt.format(**context_data) if context_data else step_prompt
                
                log_entry = {
                    "step_id": step_id,
                    "type": step_type,
                    "status": "started",
                    "timestamp": datetime.utcnow().isoformat()
                }
                execution.logs.append(log_entry)
                
                # Execute step via LLM Service
                output = await llm_service.process_step(step_type, formatted_prompt)
                
                # Store output in context for future steps
                context_data[step_id] = output
                
                log_entry["status"] = "completed"
                log_entry["output"] = output
                execution.logs.append(log_entry)
            
            execution.status = "completed"
            execution.result = context_data
            execution.completed_at = datetime.utcnow()
            
        except Exception as e:
            execution.status = "failed"
            execution.logs.append({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            execution.completed_at = datetime.utcnow()
        
        return execution

workflow_engine = WorkflowEngine()
