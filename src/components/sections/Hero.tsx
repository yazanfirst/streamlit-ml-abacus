
import React from "react";
import AnimatedButton from "../ui/AnimatedButton";
import FadeIn from "../animations/FadeIn";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex flex-col justify-center pt-20 px-6 md:px-12 overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 overflow-hidden -z-10">
        <div className="absolute w-[600px] h-[600px] rounded-full bg-accent/5 -top-[300px] -right-[300px] animate-pulse-subtle" />
        <div className="absolute w-[500px] h-[500px] rounded-full bg-accent/5 -bottom-[250px] -left-[250px] animate-pulse-subtle" style={{ animationDelay: "1s" }} />
        <div className="absolute top-1/4 left-1/4 w-4 h-4 rounded-full bg-accent/20 animate-float" style={{ animationDelay: "0s" }} />
        <div className="absolute top-1/3 right-1/3 w-6 h-6 rounded-full bg-accent/10 animate-float" style={{ animationDelay: "0.5s" }} />
        <div className="absolute bottom-1/4 right-1/4 w-5 h-5 rounded-full bg-accent/15 animate-float" style={{ animationDelay: "1s" }} />
      </div>

      <div className="max-w-7xl mx-auto w-full">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div>
            <FadeIn direction="up" delay={0.1}>
              <div className="inline-block px-3 py-1 rounded-full bg-accent/10 text-accent text-sm font-medium mb-6">
                Machine Learning, Simplified
              </div>
            </FadeIn>
            
            <FadeIn direction="up" delay={0.2}>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight leading-tight mb-6">
                Transform Your Data with <span className="text-accent">ML Abacus</span>
              </h1>
            </FadeIn>
            
            <FadeIn direction="up" delay={0.3}>
              <p className="text-lg text-muted-foreground mb-8 max-w-lg">
                A powerful yet elegant platform for data analysis and machine learning. Upload your 
                data, clean it, analyze it, and build models with just a few clicks.
              </p>
            </FadeIn>
            
            <FadeIn direction="up" delay={0.4}>
              <div className="flex flex-wrap gap-4">
                <AnimatedButton
                  variant="primary"
                  size="lg"
                  href="/app"
                >
                  Get Started
                </AnimatedButton>
                <AnimatedButton
                  variant="outline"
                  size="lg"
                  href="#features"
                >
                  Learn More
                </AnimatedButton>
              </div>
            </FadeIn>
            
            <FadeIn direction="up" delay={0.5}>
              <div className="mt-12 flex items-center">
                <div className="flex -space-x-2">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className="w-8 h-8 rounded-full bg-muted border-2 border-background overflow-hidden">
                      <div className={`w-full h-full bg-accent/${20 + i * 10}`} />
                    </div>
                  ))}
                </div>
                <div className="ml-4">
                  <p className="text-sm text-muted-foreground">
                    Trusted by <span className="font-medium text-foreground">2000+</span> data scientists
                  </p>
                </div>
              </div>
            </FadeIn>
          </div>
          
          <FadeIn direction="left" delay={0.4}>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-tr from-accent/10 to-transparent rounded-3xl blur-2xl -z-10 animate-pulse-subtle" />
              <div className="relative bg-card border border-border rounded-xl overflow-hidden shadow-xl">
                <div className="h-8 bg-muted flex items-center px-4 border-b border-border">
                  <div className="flex space-x-2">
                    <div className="w-3 h-3 rounded-full bg-destructive/70" />
                    <div className="w-3 h-3 rounded-full bg-amber-500/70" />
                    <div className="w-3 h-3 rounded-full bg-green-500/70" />
                  </div>
                </div>
                <div className="p-4 md:p-6">
                  <div className="space-y-4">
                    <div className="h-10 bg-muted/50 rounded-md w-full animate-pulse-subtle" />
                    <div className="grid grid-cols-2 gap-4">
                      <div className="h-32 bg-muted/40 rounded-md animate-pulse-subtle" style={{ animationDelay: "0.2s" }} />
                      <div className="h-32 bg-muted/40 rounded-md animate-pulse-subtle" style={{ animationDelay: "0.4s" }} />
                    </div>
                    <div className="h-40 bg-muted/30 rounded-md animate-pulse-subtle" style={{ animationDelay: "0.6s" }} />
                    <div className="flex space-x-3">
                      <div className="h-9 bg-accent/20 rounded-md w-24 animate-pulse-subtle" style={{ animationDelay: "0.8s" }} />
                      <div className="h-9 bg-muted/40 rounded-md w-24 animate-pulse-subtle" style={{ animationDelay: "1s" }} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </FadeIn>
        </div>
      </div>
    </section>
  );
};

export default Hero;
