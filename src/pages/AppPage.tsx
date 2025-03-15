
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import AnimatedButton from "@/components/ui/AnimatedButton";
import { toast } from "@/components/ui/sonner";

const AppPage = () => {
  const navigate = useNavigate();
  const [isStreamlitReady, setIsStreamlitReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  
  // Function to check if Streamlit server is running
  const checkStreamlitServer = async () => {
    try {
      const response = await fetch('http://localhost:8501/healthz');
      if (response.ok) {
        setIsStreamlitReady(true);
        setIsLoading(false);
      } else {
        setIsStreamlitReady(false);
        setIsLoading(false);
      }
    } catch (error) {
      console.error('Failed to connect to Streamlit server:', error);
      setIsStreamlitReady(false);
      setIsLoading(false);
    }
  };
  
  useEffect(() => {
    // Check Streamlit server status when component mounts
    checkStreamlitServer();
    
    // Set up an interval to periodically check if the server becomes available
    const intervalId = setInterval(checkStreamlitServer, 5000);
    
    // Clear interval on component unmount
    return () => clearInterval(intervalId);
  }, []);
  
  // Function to start Streamlit server
  const handleStartStreamlit = () => {
    toast.info(
      "Please run the Streamlit server with the command: streamlit run src/ml_abacus_app.py",
      { duration: 10000 }
    );
  };
  
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
          
          <div className="w-full h-[800px] border border-border rounded-xl overflow-hidden relative">
            {isLoading ? (
              <div className="absolute inset-0 flex items-center justify-center bg-background">
                <div className="flex flex-col items-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                  <p className="text-muted-foreground">Checking Streamlit server status...</p>
                </div>
              </div>
            ) : isStreamlitReady ? (
              <iframe 
                src="http://localhost:8501?embed=true" 
                title="ML Abacus Streamlit App"
                className="w-full h-full"
                allow="camera;microphone"
                sandbox="allow-same-origin allow-scripts allow-forms allow-downloads"
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center bg-background">
                <div className="flex flex-col items-center text-center p-8">
                  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-destructive mb-4">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                  <h3 className="text-xl font-semibold mb-2">Streamlit Server Not Detected</h3>
                  <p className="text-muted-foreground mb-6 max-w-md">
                    The Streamlit server is not running. Please start it using the command below:
                  </p>
                  <div className="bg-muted p-3 rounded-md mb-6 text-left font-mono text-sm overflow-x-auto w-full max-w-md">
                    <code>streamlit run src/ml_abacus_app.py</code>
                  </div>
                  <AnimatedButton
                    onClick={handleStartStreamlit}
                    className="mb-4"
                  >
                    Show Instructions
                  </AnimatedButton>
                  <button 
                    onClick={checkStreamlitServer} 
                    className="text-primary underline text-sm"
                  >
                    Check Again
                  </button>
                </div>
              </div>
            )}
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
