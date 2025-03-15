
import React from "react";
import { cn } from "@/lib/utils";
import FadeIn from "../animations/FadeIn";

type CardProps = {
  children: React.ReactNode;
  className?: string;
  glassmorphic?: boolean;
  interactive?: boolean;
  bordered?: boolean;
  fadeIn?: boolean;
  delay?: number;
};

const Card = ({
  children,
  className,
  glassmorphic = false,
  interactive = false,
  bordered = false,
  fadeIn = false,
  delay = 0,
}: CardProps) => {
  const cardContent = (
    <div
      className={cn(
        "rounded-xl overflow-hidden",
        glassmorphic ? "glass-card" : "bg-card text-card-foreground",
        interactive ? "transition-all duration-300 hover:shadow-lg hover:-translate-y-1" : "",
        bordered ? "border border-border" : "",
        className
      )}
    >
      {children}
    </div>
  );

  if (fadeIn) {
    return (
      <FadeIn delay={delay}>
        {cardContent}
      </FadeIn>
    );
  }

  return cardContent;
};

export default Card;
