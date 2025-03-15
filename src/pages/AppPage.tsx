
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
          <div className="text-center py-24">
            <h1 className="text-3xl md:text-4xl font-bold mb-6">
              ML Abacus App
            </h1>
            <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
              This is a placeholder for the Streamlit ML app. In a real implementation, 
              this would be integrated with a Streamlit application that allows users to
              upload CSV files, clean data, analyze, and train ML models.
            </p>
            <div className="p-8 border border-border rounded-xl bg-card inline-block">
              <p className="text-muted-foreground">Streamlit ML application would be embedded here</p>
            </div>
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
