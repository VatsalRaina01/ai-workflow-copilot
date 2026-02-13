import React from 'react';
import { Play, Clock, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const WorkflowCard = ({ workflow, onRun }) => {
    return (
        <div className="glass-panel rounded-xl p-6 hover:border-primary/50 transition-all duration-300 group relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

            <div className="relative z-10">
                <div className="flex justify-between items-start mb-4">
                    <div>
                        <h3 className="text-lg font-semibold text-white mb-1 group-hover:text-primary transition-colors">
                            {workflow.title}
                        </h3>
                        <p className="text-sm text-muted line-clamp-2">
                            {workflow.description || 'No description provided.'}
                        </p>
                    </div>
                    <span className="px-2 py-1 rounded-full text-xs font-medium bg-white/5 text-muted border border-white/10">
                        {workflow.steps_definition?.length || 0} Steps
                    </span>
                </div>

                <div className="flex items-center justify-between mt-6">
                    <div className="flex items-center text-xs text-muted">
                        <Clock className="w-3 h-3 mr-1" />
                        {new Date(workflow.created_at).toLocaleDateString()}
                    </div>

                    <div className="flex space-x-2">
                        <Link
                            to={`/run/${workflow.id}`}
                            className="p-2 rounded-lg bg-primary/10 text-primary hover:bg-primary hover:text-white transition-all duration-300"
                            title="Run Workflow"
                        >
                            <Play className="w-4 h-4 fill-current" />
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default WorkflowCard;
