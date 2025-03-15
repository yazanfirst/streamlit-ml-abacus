
import React from "react";
import { useNavigate } from "react-router-dom";
import AnimatedButton from "@/components/ui/AnimatedButton";

const AppPage = () => {
  const navigate = useNavigate();
  
  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-background border-b border-border py-4 px-6 md:px-12">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <a href="/" className="text-xl font-semibold tracking-tight">
              ML Abacus
            </a>
            <span className="inline-block px-2 py-0.5 rounded-full bg-accent/10 text-accent text-xs font-medium">
              Beta
            </span>
          </div>
          
          <AnimatedButton
            variant="ghost"
            size="sm"
            onClick={() => navigate("/")}
          >
            Back to Home
          </AnimatedButton>
        </div>
      </header>
      
      <main className="flex-1 p-6 md:p-12">
        <div className="max-w-7xl mx-auto">
          <div className="text-center py-8">
            <h1 className="text-3xl md:text-4xl font-bold mb-6">
              ML Abacus App
            </h1>
            <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
              Upload CSV files, clean data, analyze, and train ML models with our Streamlit application.
            </p>
          </div>
          
          <div className="w-full h-[800px] border border-border rounded-xl overflow-hidden">
            <iframe 
              src="http://localhost:8501" 
              title="ML Abacus Streamlit App"
              className="w-full h-full"
            />
          </div>
          
          <div className="mt-8 text-center">
            <p className="text-sm text-muted-foreground">
              Note: To use this application, make sure to run the Streamlit server with:<br />
              <code className="bg-muted p-1 rounded">streamlit run src/ml_abacus_app.py</code>
            </p>
          </div>
        </div>
      </main>
      
      <footer className="bg-secondary/50 py-4 px-6 text-center">
        <p className="text-sm text-muted-foreground">
          © {new Date().getFullYear()} ML Abacus. All rights reserved.
        </p>
      </footer>
    </div>
  );
};

export default AppPage;
