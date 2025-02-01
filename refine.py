import { Card, CardContent } from "@/components/ui/card";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

const Slide = ({ title, content, delay }) => (
  <motion.div
    initial={{ opacity: 0, y: 50 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, delay }}
    className="mb-6"
  >
    <Card className="shadow-lg rounded-2xl border border-gray-300">
      <CardContent className="p-6">
        <h2 className="text-xl font-bold mb-4">{title}</h2>
        <p className="text-base text-gray-600">{content}</p>
      </CardContent>
    </Card>
  </motion.div>
);

const Presentation = () => {
  const slides = [
    {
      title: "Pros of Splunk: Real-Time Insights",
      content:
        "Splunk processes and indexes data in real time, allowing users to monitor and respond to events as they happen.",
      delay: 0.2,
    },
    {
      title: "Versatility",
      content:
        "Supports a wide range of data formats (structured, unstructured, and semi-structured) from multiple sources, such as logs, metrics, and network traffic.",
      delay: 0.4,
    },
    {
      title: "Advanced Analytics",
      content:
        "Splunk's Search Processing Language (SPL) allows for complex queries, making it easier to analyze data and extract meaningful insights.",
      delay: 0.6,
    },
    {
      title: "Visualization",
      content:
        "Offers rich, customizable dashboards and reports to present data visually for better understanding and decision-making.",
      delay: 0.8,
    },
    {
      title: "Scalability",
      content:
        "Handles data growth easily, making it suitable for businesses of any size, from startups to large enterprises.",
      delay: 1.0,
    },
    {
      title: "Automation and Alerting",
      content:
        "Supports automated workflows, alerts, and responses based on pre-defined thresholds or anomalies.",
      delay: 1.2,
    },
    {
      title: "Security and Compliance",
      content:
        "Widely used for security information and event management (SIEM) to monitor, detect, and respond to threats. It also helps with compliance requirements by maintaining audit trails.",
      delay: 1.4,
    },
    {
      title: "Machine Learning Integration",
      content:
        "Built-in machine learning capabilities allow for anomaly detection, trend prediction, and more advanced analytics.",
      delay: 1.6,
    },
    {
      title: "Cons of Splunk: High Cost",
      content:
        "Splunk's pricing, especially for large-scale deployments, can be very expensive due to its data volume-based licensing model.",
      delay: 1.8,
    },
    {
      title: "Complexity for Beginners",
      content:
        "The learning curve for mastering SPL and understanding the platform’s functionalities can be steep for new users.",
      delay: 2.0,
    },
    {
      title: "Resource-Intensive",
      content:
        "On-premises deployments require significant infrastructure resources and can be challenging to maintain.",
      delay: 2.2,
    },
    {
      title: "Data Size Limitation",
      content:
        "Costs escalate quickly as the volume of ingested data grows, making it less cost-effective for organizations with large data volumes.",
      delay: 2.4,
    },
    {
      title: "Customization Challenges",
      content:
        "While it is highly customizable, setting up and configuring dashboards, alerts, and integrations can require significant time and expertise.",
      delay: 2.6,
    },
    {
      title: "Dependency on Expertise",
      content:
        "Organizations often need skilled professionals to manage, optimize, and interpret Splunk's data effectively.",
      delay: 2.8,
    },
    {
      title: "Usage of Splunk: IT Operations and Monitoring",
      content:
        "Monitoring application performance, server health, troubleshooting system issues by analyzing logs and metrics, and capacity planning.",
      delay: 3.0,
    },
    {
      title: "Cybersecurity",
      content:
        "SIEM for detecting and responding to security threats, analyzing security logs for intrusion detection, and forensic analysis. Compliance reporting for standards like GDPR, HIPAA, or PCI DSS.",
      delay: 3.2,
    },
    {
      title: "Business Analytics",
      content:
        "Customer behavior analysis through website and application logs, sales trend analysis to improve business operations, and tracking key performance indicators (KPIs).",
      delay: 3.4,
    },
    {
      title: "DevOps and CI/CD",
      content:
        "Monitoring application logs during deployment to detect failures and analyzing CI/CD pipeline logs for efficiency improvements.",
      delay: 3.6,
    },
    {
      title: "IoT and Edge Analytics",
      content:
        "Collecting and analyzing data from IoT devices for predictive maintenance and real-time monitoring of connected systems.",
      delay: 3.8,
    },
    {
      title: "E-Commerce and Marketing",
      content:
        "Tracking user journeys on websites or mobile apps, detecting and preventing fraudulent transactions.",
      delay: 4.0,
    },
  ];

  return (
    <div className="p-8 bg-gradient-to-b from-blue-50 to-blue-100 min-h-screen">
      <h1 className="text-3xl font-bold text-center mb-8 text-blue-600">Splunk Overview</h1>
      <div className="space-y-4">
        {slides.map((slide, index) => (
          <Slide
            key={index}
            title={slide.title}
            content={slide.content}
            delay={slide.delay}
          />
        ))}
      </div>
      <div className="text-center mt-8">
        <Button
          className="bg-blue-500 hover:bg-blue-600 text-white rounded-lg px-6 py-3 text-lg"
          onClick={() => alert("Presentation Completed!")}
        >
          Finish Presentation
        </Button>
      </div>
    </div>
  );
};

export default Presentation;
