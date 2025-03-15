
import { useLocation, useNavigate } from "react-router-dom";
import { useEffect } from "react";
import AnimatedButton from "@/components/ui/AnimatedButton";
import FadeIn from "@/components/animations/FadeIn";

const NotFound = () => {
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-background px-6">
      <div className="text-center max-w-md">
        <FadeIn direction="up">
          <div className="inline-flex items-center justify-center w-24 h-24 bg-muted rounded-full mb-8">
            <span className="text-4xl font-bold">404</span>
          </div>
          <h1 className="text-3xl font-bold mb-4">Page Not Found</h1>
          <p className="text-muted-foreground mb-8">
            The page you are looking for doesn't exist or has been moved.
          </p>
          <AnimatedButton
            variant="primary"
            onClick={() => navigate("/")}
          >
            Return to Home
          </AnimatedButton>
        </FadeIn>
      </div>
    </div>
  );
};

export default NotFound;
