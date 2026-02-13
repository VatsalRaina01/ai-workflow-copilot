import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Play, CheckCircle2, Circle, Loader2, RefreshCw, AlertCircle } from 'lucide-react';
import { getWorkflow, executeWorkflow, getExecutionStatus } from '../services/api';
import ReactMarkdown from 'react-markdown';

const ExecutionViewer = () => {
    const { id } = useParams();
    const [workflow, setWorkflow] = useState(null);
    const [execution, setExecution] = useState(null);
    const [loading, setLoading] = useState(false);
    const [polling, setPolling] = useState(false);

    useEffect(() => {
        const fetchWorkflow = async () => {
            try {
                const data = await getWorkflow(id);
                setWorkflow(data);
            } catch (error) {
                console.error("Failed to load workflow:", error);
            }
        };
        fetchWorkflow();
    }, [id]);

    useEffect(() => {
        let interval;
        if (polling && execution?.id) {
            interval = setInterval(async () => {
                try {
                    const status = await getExecutionStatus(execution.id);
                    setExecution(status);
                    if (status.status === 'completed' || status.status === 'failed') {
                        setPolling(false);
                    }
                } catch (error) {
                    console.error("Polling error:", error);
                    setPolling(false);
                }
            }, 2000); // Poll every 2 seconds
        }
        return () => clearInterval(interval);
    }, [polling, execution?.id]);

    const handleRun = async () => {
        setLoading(true);
        try {
            const result = await executeWorkflow(id);
            setExecution(result);
            setPolling(true);
        } catch (error) {
            console.error("Execution failed:", error);
        } finally {
            setLoading(false);
        }
    };

    if (!workflow) return <div className="text-center text-muted mt-20">Loading workflow...</div>;

    return (
        <div className="max-w-4xl mx-auto space-y-8">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">{workflow.title}</h1>
                    <p className="text-muted">{workflow.description}</p>
                </div>
                <button
                    onClick={handleRun}
                    disabled={loading || polling}
                    className="flex items-center px-6 py-3 rounded-xl bg-primary text-white font-semibold shadow-lg shadow-primary/25 hover:bg-primary/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {loading || polling ? (
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    ) : (
                        <Play className="w-5 h-5 mr-2" />
                    )}
                    {polling ? 'Running...' : 'Run Workflow'}
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Steps List */}
                <div className="lg:col-span-1 space-y-4">
                    <h2 className="text-lg font-semibold text-white">Execution Steps</h2>
                    <div className="space-y-3">
                        {workflow.steps_definition.map((step, index) => {
                            // Determine status based on execution logs
                            const stepLog = execution?.logs?.find(log => log.step_id === step.id && log.status === 'completed');
                            const isStarted = execution?.logs?.some(log => log.step_id === step.id);
                            const isCurrent = isStarted && !stepLog;

                            return (
                                <div
                                    key={index}
                                    className={`p-4 rounded-lg border transition-all ${stepLog
                                            ? 'bg-green-500/10 border-green-500/20'
                                            : isCurrent
                                                ? 'bg-primary/10 border-primary/20 animate-pulse'
                                                : 'bg-surface/50 border-border'
                                        }`}
                                >
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm font-medium text-white">Step {index + 1}</span>
                                        {stepLog ? (
                                            <CheckCircle2 className="w-4 h-4 text-green-500" />
                                        ) : isCurrent ? (
                                            <Loader2 className="w-4 h-4 text-primary animate-spin" />
                                        ) : (
                                            <Circle className="w-4 h-4 text-muted" />
                                        )}
                                    </div>
                                    <p className="text-xs text-muted font-mono">{step.type}</p>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Results View */}
                <div className="lg:col-span-2 space-y-6">
                    <h2 className="text-lg font-semibold text-white">Output</h2>

                    {!execution ? (
                        <div className="glass-panel p-12 rounded-xl flex flex-col items-center justify-center text-muted">
                            <Play className="w-12 h-12 mb-4 opacity-20" />
                            <p>Click "Run Workflow" to start execution</p>
                        </div>
                    ) : (
                        <div className="space-y-6">
                            {execution.logs?.filter(l => l.status === 'completed').map((log, index) => (
                                <div key={index} className="glass-panel p-6 rounded-xl animate-in fade-in slide-in-from-bottom-4 duration-500">
                                    <div className="flex items-center justify-between mb-4 border-b border-white/5 pb-4">
                                        <span className="text-sm font-medium text-primary uppercase tracking-wider">{log.type}</span>
                                        <span className="text-xs text-muted">{new Date(log.timestamp).toLocaleTimeString()}</span>
                                    </div>
                                    <div className="prose prose-invert prose-sm max-w-none">
                                        <ReactMarkdown>{log.output}</ReactMarkdown>
                                    </div>
                                </div>
                            ))}

                            {execution.status === 'failed' && (
                                <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 flex items-start text-red-400">
                                    <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0 mt-0.5" />
                                    <div>
                                        <h4 className="font-medium mb-1">Execution Failed</h4>
                                        <p className="text-sm opacity-90">An error occurred while running the workflow.</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ExecutionViewer;
