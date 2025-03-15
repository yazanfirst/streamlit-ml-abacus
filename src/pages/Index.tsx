
import React from "react";
import Navbar from "@/components/layout/Navbar";
import Hero from "@/components/sections/Hero";
import Features from "@/components/sections/Features";
import Showcase from "@/components/sections/Showcase";
import Footer from "@/components/layout/Footer";
import AnimatedButton from "@/components/ui/AnimatedButton";
import FadeIn from "@/components/animations/FadeIn";

const Index = () => {
  return (
    <div className="min-h-screen">
      <Navbar />
      <Hero />
      <Features />
      <Showcase />
      
      {/* Call to Action */}
      <section className="py-24 px-6 md:px-12 bg-accent/5">
        <div className="max-w-4xl mx-auto text-center">
          <FadeIn>
            <h2 className="text-3xl md:text-4xl font-bold mb-6">
              Ready to transform your data?
            </h2>
            <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
              Get started with ML Abacus today and discover insights from your data with our 
              elegant machine learning platform.
            </p>
            <AnimatedButton
              variant="primary"
              size="lg"
              href="/app"
            >
              Launch ML Abacus
            </AnimatedButton>
          </FadeIn>
        </div>
      </section>
      
      <Footer />
    </div>
  );
};

export default Index;
