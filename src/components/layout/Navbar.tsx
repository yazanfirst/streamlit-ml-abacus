
import React, { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import AnimatedButton from "../ui/AnimatedButton";
import FadeIn from "../animations/FadeIn";

const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 20;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [scrolled]);

  return (
    <nav
      className={cn(
        "fixed top-0 left-0 right-0 z-50 py-4 px-6 md:px-12 transition-all duration-300",
        scrolled
          ? "bg-background/80 backdrop-blur-md shadow-sm"
          : "bg-transparent"
      )}
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <FadeIn direction="down" duration={0.6}>
          <a href="/" className="text-xl font-semibold tracking-tight">
            ML Abacus
          </a>
        </FadeIn>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-8">
          <FadeIn direction="down" duration={0.6} delay={0.1}>
            <NavLink href="#features">Features</NavLink>
          </FadeIn>
          <FadeIn direction="down" duration={0.6} delay={0.2}>
            <NavLink href="#showcase">Showcase</NavLink>
          </FadeIn>
          <FadeIn direction="down" duration={0.6} delay={0.3}>
            <NavLink href="#about">About</NavLink>
          </FadeIn>
          <FadeIn direction="down" duration={0.6} delay={0.4}>
            <AnimatedButton
              variant="primary"
              size="sm"
              className="ml-2"
              href="/app"
            >
              Launch App
            </AnimatedButton>
          </FadeIn>
        </div>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden focus:outline-none"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <div className="relative w-6 h-5">
            <span
              className={cn(
                "absolute h-0.5 w-6 bg-foreground transform transition-all duration-300",
                menuOpen ? "rotate-45 top-2" : "top-0"
              )}
            />
            <span
              className={cn(
                "absolute h-0.5 bg-foreground transform transition-all duration-300",
                menuOpen ? "opacity-0 w-0" : "opacity-100 w-6 top-2"
              )}
            />
            <span
              className={cn(
                "absolute h-0.5 w-6 bg-foreground transform transition-all duration-300",
                menuOpen ? "-rotate-45 top-2" : "top-4"
              )}
            />
          </div>
        </button>
      </div>

      {/* Mobile Navigation */}
      <div
        className={cn(
          "md:hidden absolute left-0 right-0 px-6 py-4 bg-background/95 backdrop-blur-md shadow-md transition-all duration-300 ease-out-expo",
          menuOpen
            ? "opacity-100 translate-y-0 pointer-events-auto"
            : "opacity-0 -translate-y-4 pointer-events-none"
        )}
      >
        <div className="flex flex-col space-y-4">
          <NavLink href="#features" mobile onClick={() => setMenuOpen(false)}>
            Features
          </NavLink>
          <NavLink href="#showcase" mobile onClick={() => setMenuOpen(false)}>
            Showcase
          </NavLink>
          <NavLink href="#about" mobile onClick={() => setMenuOpen(false)}>
            About
          </NavLink>
          <AnimatedButton
            variant="primary"
            size="sm"
            fullWidth
            href="/app"
            onClick={() => setMenuOpen(false)}
          >
            Launch App
          </AnimatedButton>
        </div>
      </div>
    </nav>
  );
};

type NavLinkProps = {
  href: string;
  children: React.ReactNode;
  mobile?: boolean;
  onClick?: () => void;
};

const NavLink = ({ href, children, mobile, onClick }: NavLinkProps) => {
  return (
    <a
      href={href}
      className={cn(
        "transition-all duration-300 hover:text-accent relative group",
        mobile ? "text-foreground py-2" : "text-foreground/90"
      )}
      onClick={onClick}
    >
      {children}
      <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-accent transition-all duration-300 group-hover:w-full" />
    </a>
  );
};

export default Navbar;
