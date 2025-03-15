
import React, { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

type FadeInProps = {
  children: React.ReactNode;
  direction?: "up" | "down" | "left" | "right" | "none";
  delay?: number;
  duration?: number;
  className?: string;
  once?: boolean;
  threshold?: number;
};

const FadeIn = ({
  children,
  direction = "up",
  delay = 0,
  duration = 0.5,
  className,
  once = true,
  threshold = 0.1,
}: FadeInProps) => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const current = ref.current;
    if (!current) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          if (once) observer.unobserve(current);
        } else if (!once) {
          setIsVisible(false);
        }
      },
      { threshold }
    );

    observer.observe(current);

    return () => {
      if (current) observer.unobserve(current);
    };
  }, [once, threshold]);

  const getAnimation = () => {
    switch (direction) {
      case "up":
        return "animate-fade-in-up";
      case "down":
        return "animate-fade-in-down";
      case "left":
        return "animate-slide-in-left";
      case "right":
        return "animate-slide-in-right";
      case "none":
        return "animate-fade-in";
      default:
        return "animate-fade-in-up";
    }
  };

  return (
    <div
      ref={ref}
      className={cn(className)}
      style={{
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? "none" : getInitialTransform(),
        transition: `opacity ${duration}s ease-out, transform ${duration}s cubic-bezier(0.19, 1, 0.22, 1)`,
        transitionDelay: `${delay}s`,
      }}
    >
      {children}
    </div>
  );

  function getInitialTransform() {
    switch (direction) {
      case "up":
        return "translateY(20px)";
      case "down":
        return "translateY(-20px)";
      case "left":
        return "translateX(20px)";
      case "right":
        return "translateX(-20px)";
      case "none":
        return "none";
      default:
        return "translateY(20px)";
    }
  }
};

export default FadeIn;
