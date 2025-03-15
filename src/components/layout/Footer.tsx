
import React from "react";
import FadeIn from "../animations/FadeIn";

const Footer = () => {
  return (
    <footer className="py-12 px-6 md:px-12 bg-secondary/50">
      <div className="max-w-7xl mx-auto">
        <FadeIn>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="md:col-span-2">
              <h3 className="text-lg font-semibold mb-4">ML Abacus</h3>
              <p className="text-muted-foreground max-w-md">
                A powerful yet elegant machine learning platform designed for data scientists, 
                analysts, and ML enthusiasts. Transform your data into insights with our
                intuitive interface.
              </p>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-4">
                Resources
              </h4>
              <ul className="space-y-2">
                <FooterLink href="#">Documentation</FooterLink>
                <FooterLink href="#">Tutorials</FooterLink>
                <FooterLink href="#">Blog</FooterLink>
                <FooterLink href="#">API Reference</FooterLink>
              </ul>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-4">
                Company
              </h4>
              <ul className="space-y-2">
                <FooterLink href="#">About Us</FooterLink>
                <FooterLink href="#">Careers</FooterLink>
                <FooterLink href="#">Contact</FooterLink>
                <FooterLink href="#">Privacy Policy</FooterLink>
              </ul>
            </div>
          </div>
        </FadeIn>
        
        <FadeIn delay={0.2}>
          <div className="border-t border-border mt-10 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm text-muted-foreground">
              © {new Date().getFullYear()} ML Abacus. All rights reserved.
            </p>
            <div className="flex space-x-4 mt-4 md:mt-0">
              <SocialLink href="#" label="Twitter">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="w-4 h-4"
                >
                  <path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z" />
                </svg>
              </SocialLink>
              <SocialLink href="#" label="GitHub">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="w-4 h-4"
                >
                  <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />
                </svg>
              </SocialLink>
              <SocialLink href="#" label="LinkedIn">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="w-4 h-4"
                >
                  <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
                  <rect x="2" y="9" width="4" height="12" />
                  <circle cx="4" cy="4" r="2" />
                </svg>
              </SocialLink>
            </div>
          </div>
        </FadeIn>
      </div>
    </footer>
  );
};

type FooterLinkProps = {
  href: string;
  children: React.ReactNode;
};

const FooterLink = ({ href, children }: FooterLinkProps) => {
  return (
    <li>
      <a
        href={href}
        className="text-muted-foreground hover:text-foreground transition-colors duration-200"
      >
        {children}
      </a>
    </li>
  );
};

type SocialLinkProps = {
  href: string;
  label: string;
  children: React.ReactNode;
};

const SocialLink = ({ href, label, children }: SocialLinkProps) => {
  return (
    <a
      href={href}
      aria-label={label}
      className="text-muted-foreground hover:text-foreground transition-colors duration-200"
    >
      {children}
    </a>
  );
};

export default Footer;
