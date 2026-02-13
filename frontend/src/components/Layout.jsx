import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Bot, Plus, LayoutDashboard } from 'lucide-react';

const Layout = ({ children }) => {
    const location = useLocation();

    return (
        <div className="min-h-screen bg-background text-text font-sans selection:bg-primary/30">
            {/* Navigation */}
            <nav className="border-b border-border bg-surface/50 backdrop-blur-md sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-16">
                        <Link to="/" className="flex items-center space-x-3 group">
                            <div className="p-2 bg-gradient-to-tr from-primary to-secondary rounded-lg shadow-lg group-hover:shadow-primary/25 transition-all duration-300">
                                <Bot className="w-6 h-6 text-white" />
                            </div>
                            <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-white/70">
                                Workflow<span className="text-primary">Copilot</span>
                            </span>
                        </Link>

                        <div className="flex space-x-4">
                            <Link
                                to="/"
                                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${location.pathname === '/'
                                        ? 'bg-primary/10 text-primary'
                                        : 'text-muted hover:text-white hover:bg-white/5'
                                    }`}
                            >
                                <LayoutDashboard className="w-4 h-4 mr-2" />
                                Dashboard
                            </Link>
                            <Link
                                to="/create"
                                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${location.pathname === '/create'
                                        ? 'bg-primary/10 text-primary'
                                        : 'text-muted hover:text-white hover:bg-white/5'
                                    }`}
                            >
                                <Plus className="w-4 h-4 mr-2" />
                                New Workflow
                            </Link>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {children}
            </main>
        </div>
    );
};

export default Layout;
