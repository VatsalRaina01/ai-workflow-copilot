import React, { useState } from 'react';
import { Plus, Trash2, Save, ArrowRight, Wand2 } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { createWorkflow } from '../services/api';

const WorkflowBuilder = () => {
    const navigate = useNavigate();
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');
    const [steps, setSteps] = useState([
        { id: 'step_1', type: 'generation', prompt: '' }
    ]);
    const [loading, setLoading] = useState(false);

    const addStep = () => {
        const newStep = {
            id: `step_${steps.length + 1}`,
            type: 'generation',
            prompt: ''
        };
        setSteps([...steps, newStep]);
    };

    const removeStep = (index) => {
        const newSteps = steps.filter((_, i) => i !== index);
        setSteps(newSteps);
    };

    const updateStep = (index, field, value) => {
        const newSteps = [...steps];
        newSteps[index][field] = value;
        setSteps(newSteps);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            await createWorkflow({
                title,
                description,
                steps_definition: steps
            });
            navigate('/');
        } catch (error) {
            console.error("Failed to create workflow:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto">
            <div className="flex items-center justify-between mb-8">
                <h1 className="text-3xl font-bold text-white">Create New Workflow</h1>
                <Link to="/" className="text-muted hover:text-white transition-colors">Cancel</Link>
            </div>

            <form onSubmit={handleSubmit} className="space-y-8">
                {/* Basic Info */}
                <div className="glass-panel p-6 rounded-xl space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-muted mb-1">Workflow Title</label>
                        <input
                            type="text"
                            required
                            value={title}
                            onChange={(e) => setTitle(e.target.value)}
                            className="w-full bg-black/20 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary/50 transition-colors"
                            placeholder="e.g., Blog Post Generator"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-muted mb-1">Description</label>
                        <textarea
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            className="w-full bg-black/20 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary/50 transition-colors h-24 resize-none"
                            placeholder="What does this workflow do?"
                        />
                    </div>
                </div>

                {/* Steps Builder */}
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <h2 className="text-xl font-semibold text-white">Workflow Steps</h2>
                        <button
                            type="button"
                            onClick={addStep}
                            className="flex items-center px-3 py-1.5 rounded-lg bg-primary/10 text-primary hover:bg-primary hover:text-white transition-all text-sm font-medium"
                        >
                            <Plus className="w-4 h-4 mr-1" />
                            Add Step
                        </button>
                    </div>

                    <div className="space-y-4">
                        {steps.map((step, index) => (
                            <div key={index} className="glass-panel p-6 rounded-xl relative group">
                                <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button
                                        type="button"
                                        onClick={() => removeStep(index)}
                                        className="p-2 text-muted hover:text-red-400 transition-colors"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                    </button>
                                </div>

                                <div className="flex items-start space-x-4">
                                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center font-bold text-sm">
                                        {index + 1}
                                    </div>

                                    <div className="flex-1 space-y-4">
                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <label className="block text-xs font-medium text-muted mb-1">Step ID</label>
                                                <input
                                                    type="text"
                                                    value={step.id}
                                                    onChange={(e) => updateStep(index, 'id', e.target.value)}
                                                    className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-primary/50"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-xs font-medium text-muted mb-1">Type</label>
                                                <select
                                                    value={step.type}
                                                    onChange={(e) => updateStep(index, 'type', e.target.value)}
                                                    className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-primary/50"
                                                >
                                                    <option value="generation">Text Generation</option>
                                                    <option value="summary">Summarization</option>
                                                    <option value="analysis">Data Analysis</option>
                                                </select>
                                            </div>
                                        </div>

                                        <div>
                                            <label className="block text-xs font-medium text-muted mb-1">Prompt Template</label>
                                            <textarea
                                                value={step.prompt}
                                                onChange={(e) => updateStep(index, 'prompt', e.target.value)}
                                                className="w-full bg-black/20 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary/50 h-24 resize-none text-sm font-mono"
                                                placeholder="Enter prompt... Use {previous_step_id} for chaining."
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="flex justify-end pt-4">
                    <button
                        type="submit"
                        disabled={loading}
                        className="flex items-center px-6 py-3 rounded-xl bg-gradient-to-r from-primary to-secondary text-white font-semibold shadow-lg shadow-primary/25 hover:shadow-primary/40 hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? (
                            <Wand2 className="w-5 h-5 mr-2 animate-spin" />
                        ) : (
                            <Save className="w-5 h-5 mr-2" />
                        )}
                        {loading ? 'Creating...' : 'Create Workflow'}
                    </button>
                </div>
            </form>
        </div>
    );
};

export default WorkflowBuilder;
