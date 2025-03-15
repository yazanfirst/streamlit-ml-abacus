
import React from "react";
import FadeIn from "../animations/FadeIn";
import CustomCard from "../ui/CustomCard";

const showcaseSteps = [
  {
    title: "Upload & Preview Data",
    description: "Upload your CSV file and instantly preview your data to ensure it's correctly formatted.",
    image: "step1.png",
  },
  {
    title: "Clean & Preprocess",
    description: "Automatically handle missing values and encode categorical features for analysis.",
    image: "step2.png",
  },
  {
    title: "Explore & Visualize",
    description: "Generate insightful visualizations to understand relationships in your data.",
    image: "step3.png",
  },
  {
    title: "Train & Evaluate Models",
    description: "Select your model, tune parameters, and evaluate performance with detailed metrics.",
    image: "step4.png",
  },
];

const Showcase = () => {
  return (
    <section id="showcase" className="py-24 px-6 md:px-12">
      <div className="max-w-7xl mx-auto">
        <FadeIn direction="up">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              How ML Abacus Works
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              A seamless workflow from data upload to model evaluation, all in one elegant interface.
            </p>
          </div>
        </FadeIn>

        <div className="space-y-20">
          {showcaseSteps.map((step, index) => (
            <FadeIn key={index} direction={index % 2 === 0 ? "right" : "left"}>
              <div className={`grid grid-cols-1 lg:grid-cols-2 gap-12 items-center ${index % 2 === 0 ? "" : "lg:flex-row-reverse"}`}>
                <div className={index % 2 === 0 ? "lg:order-1" : "lg:order-2"}>
                  <div className="inline-block px-3 py-1 rounded-full bg-secondary text-muted-foreground text-sm font-medium mb-4">
                    Step {index + 1}
                  </div>
                  <h3 className="text-2xl font-bold mb-4">{step.title}</h3>
                  <p className="text-muted-foreground mb-6">{step.description}</p>
                  
                  <div className="border border-border rounded-lg overflow-hidden">
                    <div className="bg-muted px-4 py-2 border-b border-border flex items-center space-x-1">
                      <div className="w-3 h-3 rounded-full bg-destructive/70"></div>
                      <div className="w-3 h-3 rounded-full bg-amber-500/70"></div>
                      <div className="w-3 h-3 rounded-full bg-green-500/70"></div>
                    </div>
                    <div className="p-4 bg-card">
                      <div className="space-y-3">
                        <div className="h-4 bg-muted rounded w-full"></div>
                        <div className="h-4 bg-muted rounded w-3/4"></div>
                        <div className="h-4 bg-muted rounded w-5/6"></div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className={index % 2 === 0 ? "lg:order-2" : "lg:order-1"}>
                  <CustomCard className="overflow-hidden" glassmorphic>
                    <div className="relative aspect-video bg-muted">
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-16 h-16 rounded-full bg-secondary flex items-center justify-center">
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6">
                            <polygon points="5 3 19 12 5 21 5 3"></polygon>
                          </svg>
                        </div>
                      </div>
                      <div className="absolute inset-0 bg-gradient-to-br from-accent/10 to-primary/5"></div>
                    </div>
                  </CustomCard>
                </div>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Showcase;
