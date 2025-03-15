
import React from "react";
import { cn } from "@/lib/utils";

type AnimatedButtonProps = {
  children: React.ReactNode;
  onClick?: () => void;
  className?: string;
  variant?: "primary" | "secondary" | "ghost" | "outline";
  size?: "sm" | "md" | "lg";
  disabled?: boolean;
  type?: "button" | "submit" | "reset";
  icon?: React.ReactNode;
  iconPosition?: "left" | "right";
  loading?: boolean;
  fullWidth?: boolean;
  href?: string;
};

const AnimatedButton = ({
  children,
  onClick,
  className = "",
  variant = "primary",
  size = "md",
  disabled = false,
  type = "button",
  icon,
  iconPosition = "left",
  loading = false,
  fullWidth = false,
  href,
  ...props
}: AnimatedButtonProps) => {
  const baseClasses = "relative overflow-hidden rounded-lg font-medium transition-all duration-300 inline-flex items-center justify-center";
  
  const variantClasses = {
    primary: "bg-primary text-primary-foreground hover:opacity-90 active:scale-[0.98]",
    secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80 active:scale-[0.98]",
    ghost: "bg-transparent hover:bg-secondary text-foreground active:scale-[0.98]",
    outline: "bg-transparent border border-input text-foreground hover:bg-secondary active:scale-[0.98]",
  };
  
  const sizeClasses = {
    sm: "py-1.5 px-3 text-sm",
    md: "py-2 px-4 text-base",
    lg: "py-2.5 px-5 text-lg",
  };

  const buttonClasses = cn(
    baseClasses,
    variantClasses[variant],
    sizeClasses[size],
    fullWidth ? "w-full" : "",
    disabled || loading ? "opacity-60 cursor-not-allowed" : "cursor-pointer",
    className
  );

  const renderContent = () => (
    <>
      {loading ? (
        <span className="animate-rotate-slow inline-block h-4 w-4 border-2 border-current border-r-transparent rounded-full mr-2" />
      ) : icon && iconPosition === "left" ? (
        <span className="mr-2">{icon}</span>
      ) : null}
      <span className="relative z-10">{children}</span>
      {icon && iconPosition === "right" && !loading ? (
        <span className="ml-2">{icon}</span>
      ) : null}
      <span className="absolute inset-0 w-full h-full bg-white/10 opacity-0 hover:opacity-100 transition-opacity duration-300" />
    </>
  );

  if (href && !disabled) {
    return (
      <a href={href} className={buttonClasses} {...props}>
        {renderContent()}
      </a>
    );
  }

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={buttonClasses}
      {...props}
    >
      {renderContent()}
    </button>
  );
};

export default AnimatedButton;
