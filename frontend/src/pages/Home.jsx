import React, { useEffect, useState } from 'react';
import Layout from '../components/Layout';
import WorkflowCard from '../components/WorkflowCard';
import { getWorkflows } from '../services/api';
import { Plus, Loader2 } from 'lucide-react';
import { Link } from 'react-router-dom';

const Home = () => {
    const [workflows, setWorkflows] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchWorkflows = async () => {
            try {
                const data = await getWorkflows();
                setWorkflows(data);
            } catch (error) {
                console.error("Failed to fetch workflows:", error);
            } finally {
                setLoading(false);
            }
        };
        fetchWorkflows();
    }, []);

    return (
        <Layout>
            <div className="space-y-8">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                        <h1 className="text-4xl font-bold text-white mb-2">
                            <span className="gradient-text">AI Workflows</span>
                        </h1>
                        <p className="text-muted text-lg">Automate your tasks with intelligent agentic workflows.</p>
                    </div>

                    <Link
                        to="/create"
                        className="flex items-center justify-center px-6 py-3 rounded-xl bg-white text-black font-semibold hover:bg-white/90 transition-all hover:scale-105"
                    >
                        <Plus className="w-5 h-5 mr-2" />
                        New Workflow
                    </Link>
                </div>

                {loading ? (
                    <div className="flex justify-center items-center py-20">
                        <Loader2 className="w-8 h-8 text-primary animate-spin" />
                    </div>
                ) : workflows.length === 0 ? (
                    <div className="glass-panel rounded-2xl p-12 text-center">
                        <div className="w-16 h-16 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-6">
                            <Plus className="w-8 h-8 text-muted" />
                        </div>
                        <h3 className="text-xl font-semibold text-white mb-2">No workflows yet</h3>
                        <p className="text-muted mb-8 max-w-md mx-auto">
                            Create your first AI workflow to start automating your tasks using powerful language models.
                        </p>
                        <Link
                            to="/create"
                            className="inline-flex items-center px-6 py-3 rounded-xl bg-primary text-white font-semibold hover:bg-primary/90 transition-all"
                        >
                            Create Workflow
                        </Link>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {workflows.map((workflow) => (
                            <WorkflowCard key={workflow.id} workflow={workflow} />
                        ))}
                    </div>
                )}
            </div>
        </Layout>
    );
};

export default Home;
