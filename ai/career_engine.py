"""
ai/career_engine.py
AI-driven Career Recommendation and Guidance Engine.
Combines RIASEC Holland Code personality matching, multi-dimensional aptitude scoring,
and scikit-learn cosine similarity for personalized career suggestions, skill gap analysis,
and structured 4-phase career roadmaps.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# CAREER KNOWLEDGE BASE
# ---------------------------------------------------------------------------

CAREER_DATABASE: Dict[str, Dict[str, Any]] = {
    "software_engineer": {
        "title": "Software Engineer / Full Stack Developer",
        "category": "Technology & Computer Science",
        "description": "Design, build, and maintain scalable web and desktop applications, APIs, and software architectures.",
        # Vector: [Numerical, Logical, Verbal, Technical, R, I, A, S, E, C]
        "vector": [0.85, 0.95, 0.65, 0.95, 0.6, 0.9, 0.4, 0.4, 0.5, 0.8],
        "required_skills": ["Python", "JavaScript", "SQL", "Git", "Data Structures", "System Design", "REST APIs"],
        "soft_skills": ["Problem Solving", "Attention to Detail", "Team Collaboration", "Continuous Learning"],
        "avg_salary_inr": "₹7,00,000 - ₹22,00,000 / year",
        "market_demand": "High Demand (Expected +25% growth by 2030)",
        "roadmap": [
            {
                "phase": "Phase 1: Foundations",
                "topics": "Programming Fundamentals (Python / JavaScript), Object-Oriented Programming, Basic CLI & Git"
            },
            {
                "phase": "Phase 2: Core Development",
                "topics": "Relational Databases (SQL), Web Frameworks (Flask/Django or Node.js), Frontend Basics (HTML/CSS/JS)"
            },
            {
                "phase": "Phase 3: Advanced Systems",
                "topics": "Data Structures & Algorithms, RESTful APIs, Containerization (Docker), Automated Testing"
            },
            {
                "phase": "Phase 4: Industry Readiness",
                "topics": "Cloud Deployment (AWS/GCP), CI/CD Pipelines, Open-source contributions, Full-Stack Capstone Project"
            }
        ],
        "resources": [
            {"name": "roadmap.sh - Full Stack Developer", "type": "Interactive Guide", "url": "https://roadmap.sh/full-stack"},
            {"name": "freeCodeCamp Web Development", "type": "Free Certification", "url": "https://www.freecodecamp.org/"},
            {"name": "CS50: Introduction to Computer Science (Harvard)", "type": "Course", "url": "https://pll.harvard.edu/course/cs50"}
        ]
    },

    "ai_ml_engineer": {
        "title": "AI & Machine Learning Engineer",
        "category": "Technology & Computer Science",
        "description": "Develop and deploy predictive models, neural networks, and generative AI solutions to solve complex problems.",
        "vector": [0.95, 0.95, 0.70, 0.90, 0.4, 0.95, 0.4, 0.3, 0.5, 0.7],
        "required_skills": ["Python", "Machine Learning", "Deep Learning", "NumPy & Pandas", "PyTorch / TensorFlow", "Mathematics & Statistics"],
        "soft_skills": ["Analytical Mindset", "Research Curiosity", "Algorithmic Thinking", "Data Ethics"],
        "avg_salary_inr": "₹9,00,000 - ₹28,00,000 / year",
        "market_demand": "Very High Demand (+35% growth, Generative AI boom)",
        "roadmap": [
            {
                "phase": "Phase 1: Math & Foundations",
                "topics": "Linear Algebra, Multivariable Calculus, Probability & Statistics, Advanced Python Programming"
            },
            {
                "phase": "Phase 2: Classical Machine Learning",
                "topics": "Scikit-Learn, Regression, Classification, Clustering, Model Evaluation, Feature Engineering"
            },
            {
                "phase": "Phase 3: Deep Learning & NLP/CV",
                "topics": "PyTorch or TensorFlow, CNNs, Transformers, LLMs, Vector Databases, Hugging Face"
            },
            {
                "phase": "Phase 4: MLOps & Production",
                "topics": "Docker, MLflow, FastAPI Model Serving, Cloud GPU Deployment, End-to-End AI Project"
            }
        ],
        "resources": [
            {"name": "DeepLearning.AI - Machine Learning Specialization", "type": "Course", "url": "https://www.deeplearning.ai/"},
            {"name": "Kaggle Learn & Competitions", "type": "Hands-on Practice", "url": "https://www.kaggle.com/learn"},
            {"name": "Fast.ai - Practical Deep Learning", "type": "Free Course", "url": "https://course.fast.ai/"}
        ]
    },

    "data_scientist": {
        "title": "Data Scientist / Business Intelligence Analyst",
        "category": "Technology & Computer Science",
        "description": "Extract actionable business insights from big data through statistical modeling, data visualization, and storytelling.",
        "vector": [0.90, 0.90, 0.80, 0.80, 0.3, 0.90, 0.5, 0.5, 0.6, 0.8],
        "required_skills": ["Python", "SQL", "Tableau / Power BI", "Statistics", "Pandas", "Data Cleaning", "Data Visualization"],
        "soft_skills": ["Business Acumen", "Visual Storytelling", "Critical Thinking", "Stakeholder Communication"],
        "avg_salary_inr": "₹8,00,000 - ₹24,00,000 / year",
        "market_demand": "High Demand (+28% growth)",
        "roadmap": [
            {
                "phase": "Phase 1: Data Manipulation",
                "topics": "Advanced Excel, SQL Querying, Python (Pandas & NumPy), Data Cleaning"
            },
            {
                "phase": "Phase 2: Exploratory Analysis & BI",
                "topics": "Exploratory Data Analysis (EDA), Seaborn, Power BI / Tableau Dashboards, Statistical Tests"
            },
            {
                "phase": "Phase 3: Predictive Modeling",
                "topics": "Supervised & Unsupervised Machine Learning, Hypothesis Testing, A/B Testing"
            },
            {
                "phase": "Phase 4: Business Impact Portfolio",
                "topics": "End-to-End Business Case Studies, Stakeholder Presentation, Dashboard Deployment"
            }
        ],
        "resources": [
            {"name": "Google Data Analytics Professional Certificate", "type": "Certification", "url": "https://grow.google/certificates/data-analytics/"},
            {"name": "Mode Analytics SQL Tutorial", "type": "Interactive Tutorial", "url": "https://mode.com/sql-tutorial/"},
            {"name": "StatQuest with Josh Starmer", "type": "Video Series", "url": "https://statquest.org/"}
        ]
    },

    "cybersecurity_analyst": {
        "title": "Cybersecurity Analyst / Information Security Specialist",
        "category": "Technology & Computer Science",
        "description": "Protect computer systems, networks, and confidential data against unauthorized breaches and cyber threats.",
        "vector": [0.75, 0.90, 0.65, 0.90, 0.5, 0.85, 0.3, 0.4, 0.5, 0.9],
        "required_skills": ["Networking Protocols (TCP/IP)", "Linux", "Ethical Hacking", "SIEM Tools", "Cryptography", "Python/Bash Scripting"],
        "soft_skills": ["Threat Awareness", "Calm Under Pressure", "Integrity", "Investigative Curiosity"],
        "avg_salary_inr": "₹6,50,000 - ₹20,00,000 / year",
        "market_demand": "Extremely High (Global talent shortage, +33% growth)",
        "roadmap": [
            {
                "phase": "Phase 1: IT & Network Fundamentals",
                "topics": "OSI Model, TCP/IP, Linux CLI Administration, Basic Bash/Python Scripting"
            },
            {
                "phase": "Phase 2: Security Principles & Tools",
                "topics": "Wireshark, Nmap, Cryptography basics, Vulnerability Scanning, CompTIA Security+ concepts"
            },
            {
                "phase": "Phase 3: Defensive & Offensive Ops",
                "topics": "SOC operations, Splunk / Elastic SIEM, OWASP Top 10 Web Vulnerabilities, Penetration Testing basics"
            },
            {
                "phase": "Phase 4: Certifications & Labs",
                "topics": "TryHackMe / HackTheBox labs, CEH or Security+ Certification, Incident Response Simulation"
            }
        ],
        "resources": [
            {"name": "TryHackMe - Hands-on Cyber Security", "type": "Interactive Labs", "url": "https://tryhackme.com/"},
            {"name": "Cybrary Free Cybersecurity Training", "type": "Course Platform", "url": "https://www.cybrary.it/"},
            {"name": "Professor Messer Security+ Training", "type": "Free Video Course", "url": "https://www.professormesser.com/"}
        ]
    },

    "cloud_devops_architect": {
        "title": "Cloud Architect & DevOps Engineer",
        "category": "Technology & Computer Science",
        "description": "Design and manage resilient cloud infrastructure, automated CI/CD pipelines, and high-availability systems.",
        "vector": [0.80, 0.85, 0.65, 0.95, 0.6, 0.80, 0.3, 0.4, 0.6, 0.85],
        "required_skills": ["AWS / Azure / GCP", "Docker", "Kubernetes", "Linux", "Terraform (IaC)", "CI/CD (GitHub Actions/Jenkins)"],
        "soft_skills": ["Systemic Thinking", "Proactive Monitoring", "Cross-team Collaboration", "Reliability Focus"],
        "avg_salary_inr": "₹8,50,000 - ₹26,00,000 / year",
        "market_demand": "High Demand (Cloud migration across all industries)",
        "roadmap": [
            {
                "phase": "Phase 1: Linux & Networking",
                "topics": "Linux System Administration, Shell Scripting, Networking & DNS, Git workflow"
            },
            {
                "phase": "Phase 2: Cloud Core",
                "topics": "AWS / Azure core services (Compute, Storage, VPC, IAM), Cloud Practitioner / Solutions Architect concepts"
            },
            {
                "phase": "Phase 3: Containers & CI/CD",
                "topics": "Docker containerization, Kubernetes cluster orchestration, GitHub Actions CI/CD pipelines"
            },
            {
                "phase": "Phase 4: Infrastructure as Code & Monitoring",
                "topics": "Terraform, Ansible, Prometheus & Grafana monitoring, Cloud Architect Capstone project"
            }
        ],
        "resources": [
            {"name": "AWS Skill Builder", "type": "Official Training", "url": "https://explore.skillbuilder.aws/"},
            {"name": "roadmap.sh - DevOps Roadmap", "type": "Interactive Guide", "url": "https://roadmap.sh/devops"},
            {"name": "Kubernetes Official Tutorials", "type": "Documentation", "url": "https://kubernetes.io/docs/tutorials/"}
        ]
    },

    "ui_ux_designer": {
        "title": "UI/UX Product Designer",
        "category": "Design & Creative Arts",
        "description": "Craft intuitive, beautiful, and accessible user experiences for web and mobile digital products.",
        "vector": [0.45, 0.70, 0.80, 0.60, 0.3, 0.60, 0.95, 0.7, 0.5, 0.6],
        "required_skills": ["Figma", "User Research", "Wireframing & Prototyping", "Design Systems", "Usability Testing", "Information Architecture"],
        "soft_skills": ["Empathy", "Visual Aesthetics", "Communication", "Iterative Mindset"],
        "avg_salary_inr": "₹6,00,000 - ₹18,00,000 / year",
        "market_demand": "High Demand (Digital-first businesses require strong product design)",
        "roadmap": [
            {
                "phase": "Phase 1: Design Principles & Figma",
                "topics": "Color Theory, Typography, Visual Hierarchy, Figma mastery (Auto-layout, Components, Variants)"
            },
            {
                "phase": "Phase 2: User Experience (UX) Process",
                "topics": "User Personas, User Journey Mapping, Information Architecture, Low-fidelity Wireframes"
            },
            {
                "phase": "Phase 3: High-Fidelity & Prototyping",
                "topics": "Interactive Prototypes, Micro-interactions, Usability Testing, Accessibility (WCAG)"
            },
            {
                "phase": "Phase 4: Design Systems & Portfolio",
                "topics": "Building scalable Design Systems, Writing UX Case Studies, Publishing Behance/Dribbble Portfolio"
            }
        ],
        "resources": [
            {"name": "Google UX Design Professional Certificate", "type": "Certification", "url": "https://grow.google/certificates/ux-design/"},
            {"name": "Figma Learn Tutorials", "type": "Official Guides", "url": "https://help.figma.com/hc/en-us/categories/360002051613"},
            {"name": "Nielsen Norman Group UX Articles", "type": "Research Hub", "url": "https://www.nngroup.com/articles/"}
        ]
    },

    "game_designer": {
        "title": "Game Designer & Interactive Media Developer",
        "category": "Design & Creative Arts",
        "description": "Build immersive games, gameplay mechanics, level designs, and interactive storytelling experiences.",
        "vector": [0.65, 0.80, 0.65, 0.85, 0.5, 0.65, 0.95, 0.4, 0.5, 0.5],
        "required_skills": ["Unity / Unreal Engine", "C# or C++", "Game Mechanics", "Level Design", "3D Basics (Blender)", "Sound Design basics"],
        "soft_skills": ["Creative Imagination", "Narrative Storytelling", "Player Empathy", "Attention to Pacing"],
        "avg_salary_inr": "₹5,50,000 - ₹17,00,000 / year",
        "market_demand": "Steady Growth (Mobile gaming, indie studios, VR/AR spaces)",
        "roadmap": [
            {
                "phase": "Phase 1: Game Design Theory & Engines",
                "topics": "Game Loop fundamentals, Core Mechanics, Unity or Unreal Engine installation, C# basics"
            },
            {
                "phase": "Phase 2: 2D Game Development",
                "topics": "Physics, Sprites, Player Controls, Colliders, UI Canvas, Publishing a mini 2D game"
            },
            {
                "phase": "Phase 3: 3D Mechanics & Level Design",
                "topics": "3D Transformations, Lighting, NavMesh AI, Particle effects, Audio integration"
            },
            {
                "phase": "Phase 4: Game Jam & Portfolio",
                "topics": "Participate in Game Jams (Itch.io), Polish gameplay, build a playable portfolio reel"
            }
        ],
        "resources": [
            {"name": "Unity Learn Platform", "type": "Free Interactive Course", "url": "https://learn.unity.com/"},
            {"name": "Unreal Engine Online Learning", "type": "Official Tutorials", "url": "https://dev.epicgames.com/community/learning"},
            {"name": "Brackeys Game Dev Archive", "type": "Video Tutorials", "url": "https://www.youtube.com/c/Brackeys"}
        ]
    },

    "digital_product_designer": {
        "title": "Digital Product & Brand Designer",
        "category": "Design & Creative Arts",
        "description": "Shape cohesive brand identities, marketing visuals, design assets, and digital design collateral.",
        "vector": [0.40, 0.60, 0.85, 0.60, 0.3, 0.55, 0.95, 0.6, 0.6, 0.5],
        "required_skills": ["Adobe Illustrator", "Photoshop", "Brand Guidelines", "Visual Storytelling", "Typography", "Motion Design basics"],
        "soft_skills": ["Creative Direction", "Client Communication", "Aesthetic Vision", "Adaptability"],
        "avg_salary_inr": "₹5,00,000 - ₹15,00,000 / year",
        "market_demand": "High Demand (Content creator economy, tech branding)",
        "roadmap": [
            {
                "phase": "Phase 1: Graphic Design Fundamentals",
                "topics": "Color Theory, Grid Systems, Composition, Vector Graphics in Illustrator"
            },
            {
                "phase": "Phase 2: Brand Identity Systems",
                "topics": "Logo Design, Brand Books, Typography pairings, Visual identity mockups"
            },
            {
                "phase": "Phase 3: Digital & Motion Media",
                "topics": "Social Media assets, After Effects motion graphics, Advertising creative design"
            },
            {
                "phase": "Phase 4: Commercial Portfolio",
                "topics": "Client freelance projects, Case studies, Professional Behance & personal website"
            }
        ],
        "resources": [
            {"name": "Canva Design School", "type": "Free Guides", "url": "https://www.canva.com/designschool/"},
            {"name": "The Futur - Design Business & Craft", "type": "Video Channel", "url": "https://thefutur.com/"},
            {"name": "Behance Inspiration", "type": "Showcase", "url": "https://www.behance.net/"}
        ]
    },

    "product_manager": {
        "title": "Technical Product Manager",
        "category": "Business & Management",
        "description": "Lead the product vision, define roadmaps, align cross-functional engineering, design, and business teams.",
        "vector": [0.70, 0.85, 0.90, 0.70, 0.2, 0.70, 0.6, 0.8, 0.95, 0.7],
        "required_skills": ["Product Roadmapping", "Agile / Scrum", "Data Analysis (SQL/Metrics)", "User Research", "Market Sizing", "Wireframing"],
        "soft_skills": ["Leadership", "Strategic Prioritization", "Stakeholder Diplomacy", "Executive Presentation"],
        "avg_salary_inr": "₹12,00,000 - ₹32,00,000 / year",
        "market_demand": "Very High (Strategic leadership across tech companies)",
        "roadmap": [
            {
                "phase": "Phase 1: Product Fundamentals",
                "topics": "Product Lifecycle, Agile/Scrum methodologies, User Story writing, PRD (Product Requirement Document)"
            },
            {
                "phase": "Phase 2: Product Analytics",
                "topics": "Defining KPIs (DAU, Retention, CAC, LTV), A/B Testing, Mixpanel / Amplitude data tracking"
            },
            {
                "phase": "Phase 3: Strategy & Market Fit",
                "topics": "Competitive analysis, Pricing strategy, Go-To-Market (GTM) execution, Customer discovery interviews"
            },
            {
                "phase": "Phase 4: Product Leadership Portfolio",
                "topics": "Product teardowns, Case study presentations, Managing simulated product launches"
            }
        ],
        "resources": [
            {"name": "Product School Resources", "type": "Community & Guides", "url": "https://productschool.com/resources"},
            {"name": "Lenny's Newsletter & Podcast", "type": "Industry Insights", "url": "https://www.lennysnewsletter.com/"},
            {"name": "Inspired by Marty Cagan", "type": "Key Literature", "url": "https://svpg.com/books/inspired/"}
        ]
    },

    "financial_analyst": {
        "title": "Financial Analyst / Investment Specialist",
        "category": "Business & Management",
        "description": "Analyze financial data, forecast market trends, evaluate investment opportunities, and build valuation models.",
        "vector": [0.95, 0.90, 0.75, 0.50, 0.2, 0.80, 0.3, 0.5, 0.85, 0.95],
        "required_skills": ["Financial Modeling", "Excel (Advanced)", "Valuation (DCF)", "Accounting Principles", "SQL / Python for Finance", "Data Visualization"],
        "soft_skills": ["Quantitative Precision", "Critical Due Diligence", "Presentation to Investors", "Ethical Responsibility"],
        "avg_salary_inr": "₹7,00,000 - ₹21,00,000 / year",
        "market_demand": "Steady High Demand (Fintech, Banking, Corporate Finance)",
        "roadmap": [
            {
                "phase": "Phase 1: Financial & Accounting Core",
                "topics": "Balance Sheets, P&L, Cash Flow Statements, Financial ratios, Mastery of Microsoft Excel"
            },
            {
                "phase": "Phase 2: Financial Modeling & Valuation",
                "topics": "Discounted Cash Flow (DCF), Comparable Company Analysis, Three-statement dynamic models"
            },
            {
                "phase": "Phase 3: Computational Finance",
                "topics": "Python for Finance (Pandas, Time Series, Monte Carlo simulations), SQL queries for financial DBs"
            },
            {
                "phase": "Phase 4: Certifications & Deal Pitch",
                "topics": "CFA Level 1 preparation or NISM certifications, Real-world Equity Research report presentation"
            }
        ],
        "resources": [
            {"name": "Corporate Finance Institute (CFI)", "type": "Financial Modeling Courses", "url": "https://corporatefinanceinstitute.com/"},
            {"name": "Investopedia Academy", "type": "Educational Hub", "url": "https://www.investopedia.com/"},
            {"name": "Aswath Damodaran Corporate Finance Online", "type": "NYU Stern Free Lectures", "url": "https://pages.stern.nyu.edu/~adamodar/"}
        ]
    },

    "digital_marketing_strategist": {
        "title": "Digital Growth Strategist & Marketing Lead",
        "category": "Business & Management",
        "description": "Drive customer acquisition, organic search ranking (SEO), paid campaigns, and content growth funnels.",
        "vector": [0.55, 0.70, 0.90, 0.60, 0.2, 0.65, 0.75, 0.75, 0.90, 0.6],
        "required_skills": ["SEO & SEM", "Google Analytics 4", "Meta Ads Manager", "Content Marketing", "Email Automation", "Copywriting"],
        "soft_skills": ["Persuasive Communication", "Creativity", "Data Interpretation", "Market Trend Sensing"],
        "avg_salary_inr": "₹5,00,000 - ₹16,00,000 / year",
        "market_demand": "High Demand (Digital e-commerce and SaaS expansion)",
        "roadmap": [
            {
                "phase": "Phase 1: Marketing Fundamentals & Copywriting",
                "topics": "Audience segmentation, Value proposition crafting, Compelling copywriting, Social media algorithms"
            },
            {
                "phase": "Phase 2: Organic Search & Content Strategy",
                "topics": "SEO (Keyword Research, On-page, Technical SEO, Backlink strategies), Content calendar execution"
            },
            {
                "phase": "Phase 3: Performance Marketing & Ads",
                "topics": "Google Search & Display Ads, Meta Ads Manager, Conversion Rate Optimization (CRO), Google Analytics 4"
            },
            {
                "phase": "Phase 4: Funnel Automation & Growth Loops",
                "topics": "HubSpot / Klaviyo email flows, Growth Hacking experiments, Managing multi-channel campaign ROI"
            }
        ],
        "resources": [
            {"name": "Google Digital Garage", "type": "Free Certification", "url": "https://learndigital.withgoogle.com/digitalgarage"},
            {"name": "HubSpot Academy Inbound Marketing", "type": "Free Certification", "url": "https://academy.hubspot.com/"},
            {"name": "Moz Beginner's Guide to SEO", "type": "Comprehensive Guide", "url": "https://moz.com/beginners-guide-to-seo"}
        ]
    },

    "clinical_psychologist": {
        "title": "Counselor & Clinical Psychologist",
        "category": "Healthcare & Social Services",
        "description": "Assess psychological well-being, provide therapeutic guidance, and help individuals overcome mental health challenges.",
        "vector": [0.40, 0.75, 0.90, 0.30, 0.2, 0.80, 0.6, 0.98, 0.5, 0.6],
        "required_skills": ["Counseling Techniques", "Psychological Assessment", "Empathy & Active Listening", "Crisis Intervention", "Cognitive Behavioral Principles"],
        "soft_skills": ["Deep Compassion", "Emotional Resilience", "Confidentiality", "Non-judgmental Communication"],
        "avg_salary_inr": "₹4,50,000 - ₹14,00,000 / year",
        "market_demand": "High Demand (Growing global emphasis on mental health and corporate wellness)",
        "roadmap": [
            {
                "phase": "Phase 1: Undergraduate Foundations",
                "topics": "Bachelor's in Psychology, Developmental Psychology, Biological bases of behavior, Research methods"
            },
            {
                "phase": "Phase 2: Advanced Theories & Assessments",
                "topics": "Master's degree coursework, Psychopathology (DSM-5/ICD-11), Psychometric testing, Ethics in practice"
            },
            {
                "phase": "Phase 3: Supervised Clinical Internships",
                "topics": "Hospital psychiatric ward or counseling center rotations, Supervised client intake sessions"
            },
            {
                "phase": "Phase 4: Licensure & Practice Specialization",
                "topics": "Professional registration (RCI or equivalent state licensure), Workshop certifications (CBT, Mindfulness)"
            }
        ],
        "resources": [
            {"name": "American Psychological Association (APA)", "type": "Professional Hub", "url": "https://www.apa.org/"},
            {"name": "Coursera Psychological First Aid (Johns Hopkins)", "type": "Online Course", "url": "https://www.coursera.org/learn/psychological-first-aid"},
            {"name": "Psychology Today Insights", "type": "Articles & Case Studies", "url": "https://www.psychologytoday.com/"}
        ]
    },

    "biomedical_engineer": {
        "title": "Biomedical Engineer & Healthcare Technologist",
        "category": "Healthcare & Social Services",
        "description": "Design medical devices, diagnostic equipment, prosthetics, and digital healthcare technologies.",
        "vector": [0.85, 0.90, 0.65, 0.85, 0.7, 0.90, 0.3, 0.6, 0.5, 0.7],
        "required_skills": ["Biomechanics", "Medical Device Design (CAD)", "Biosensors", "Signal Processing", "FDA/Medical Regulatory Standards", "Python / MATLAB"],
        "soft_skills": ["Patient Safety Awareness", "Precision", "Multidisciplinary Collaboration", "Inventive Thinking"],
        "avg_salary_inr": "₹6,00,000 - ₹19,00,000 / year",
        "market_demand": "Growing Demand (Advancements in telemedicine, smart wearable diagnostics, robotics)",
        "roadmap": [
            {
                "phase": "Phase 1: Engineering & Biology Basics",
                "topics": "Human Anatomy & Physiology, Circuit Theory, CAD Design (SolidWorks/Fusion 360), Engineering Mathematics"
            },
            {
                "phase": "Phase 2: Biosignals & Medical Instrumentation",
                "topics": "Biomedical sensors (ECG, EMG, EEG), Signal acquisition & filtering with MATLAB/Python"
            },
            {
                "phase": "Phase 3: Biomaterials & Device Prototyping",
                "topics": "Biocompatible materials, 3D printing of prosthetics, ISO 13485 & medical device regulatory compliance"
            },
            {
                "phase": "Phase 4: Capstone & Industry Clinical Trials",
                "topics": "Functional medical device prototype development, Hospital testing collaboration, Patent documentation"
            }
        ],
        "resources": [
            {"name": "IEEE Engineering in Medicine and Biology Society", "type": "Global Community", "url": "https://www.embs.org/"},
            {"name": "MIT OpenCourseWare - Biological Engineering", "type": "Free Lectures", "url": "https://ocw.mit.edu/courses/biological-engineering/"},
            {"name": "Medical Device Academy", "type": "Regulatory Guides", "url": "https://medicaldeviceacademy.com/"}
        ]
    },

    "public_health_coordinator": {
        "title": "Public Health & Healthcare Administrator",
        "category": "Healthcare & Social Services",
        "description": "Plan community wellness initiatives, health policy campaigns, hospital operations, and healthcare access programs.",
        "vector": [0.60, 0.75, 0.85, 0.55, 0.2, 0.75, 0.4, 0.90, 0.80, 0.8],
        "required_skills": ["Epidemiology", "Healthcare Policy", "Health Informatics", "Program Management", "Biostatistics", "Community Outreach"],
        "soft_skills": ["Public Advocacy", "Crisis Organization", "Cultural Sensitivity", "Clear Public Communication"],
        "avg_salary_inr": "₹5,50,000 - ₹16,00,000 / year",
        "market_demand": "High Demand (Global health surveillance, healthcare NGO expansion)",
        "roadmap": [
            {
                "phase": "Phase 1: Public Health Fundamentals",
                "topics": "Global Health systems, Determinants of health, Basic Biostatistics, Healthcare ethics"
            },
            {
                "phase": "Phase 2: Epidemiology & Research Methods",
                "topics": "Disease tracking, Epidemiological study designs, Survey data collection, R / SPSS data analysis"
            },
            {
                "phase": "Phase 3: Health Administration & Policy",
                "topics": "Hospital administration workflows, Public health budgeting, Grant proposal writing"
            },
            {
                "phase": "Phase 4: Field Intervention & Leadership",
                "topics": "Leading community wellness campaign, Policy white paper publication, Inter-agency health coordination"
            }
        ],
        "resources": [
            {"name": "World Health Organization (WHO) Academy", "type": "Global Training", "url": "https://www.who.int/about/who-academy"},
            {"name": "CDC Learning Connection", "type": "Public Health Hub", "url": "https://www.cdc.gov/learning/"},
            {"name": "Johns Hopkins Public Health Online", "type": "Course Materials", "url": "https://publichealth.jhu.edu/"}
        ]
    }
}

# ---------------------------------------------------------------------------
# RIASEC DOMAIN MAPPING
# ---------------------------------------------------------------------------

RIASEC_NAMES = {
    "R": "Realistic (Hands-on, Practical, Technical)",
    "I": "Investigative (Analytical, Scientific, Inquisitive)",
    "A": "Artistic (Creative, Expressive, Original)",
    "S": "Social (Supportive, Educational, Empathetic)",
    "E": "Enterprising (Persuasive, Leadership, Business)",
    "C": "Conventional (Detail-oriented, Organized, Systematic)"
}

CATEGORY_MAP = {
    "Technology & Computer Science": ["software_engineer", "ai_ml_engineer", "data_scientist", "cybersecurity_analyst", "cloud_devops_architect"],
    "Design & Creative Arts": ["ui_ux_designer", "game_designer", "digital_product_designer"],
    "Business & Management": ["product_manager", "financial_analyst", "digital_marketing_strategist"],
    "Healthcare & Social Services": ["clinical_psychologist", "biomedical_engineer", "public_health_coordinator"]
}


# ---------------------------------------------------------------------------
# CORE RECOMMENDATION ENGINE
# ---------------------------------------------------------------------------

def calculate_student_vector(
    aptitude_scores: Dict[str, float],
    riasec_scores: Dict[str, float]
) -> np.ndarray:
    """
    Construct a 10-dimensional normalized vector for student:
    [Numerical, Logical, Verbal, Technical, R, I, A, S, E, C]
    Scores expected in ranges:
      aptitude: 0-10 or 0-1
      riasec: 0-10 or 0-1
    """
    # Normalize aptitude (assume max 10, or take as ratio)
    num = min(1.0, max(0.1, aptitude_scores.get("numerical", 5) / 10.0))
    log = min(1.0, max(0.1, aptitude_scores.get("logical", 5) / 10.0))
    verb = min(1.0, max(0.1, aptitude_scores.get("verbal", 5) / 10.0))
    tech = min(1.0, max(0.1, aptitude_scores.get("technical", 5) / 10.0))

    # RIASEC scores normalized
    r_val = min(1.0, max(0.1, riasec_scores.get("R", 3) / 10.0))
    i_val = min(1.0, max(0.1, riasec_scores.get("I", 3) / 10.0))
    a_val = min(1.0, max(0.1, riasec_scores.get("A", 3) / 10.0))
    s_val = min(1.0, max(0.1, riasec_scores.get("S", 3) / 10.0))
    e_val = min(1.0, max(0.1, riasec_scores.get("E", 3) / 10.0))
    c_val = min(1.0, max(0.1, riasec_scores.get("C", 3) / 10.0))

    vec = np.array([num, log, verb, tech, r_val, i_val, a_val, s_val, e_val, c_val]).reshape(1, -1)
    return vec


def calculate_skill_gap(career_info: Dict[str, Any], student_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate matching skills and missing skills given student skills."""
    if not student_skills:
        student_skills = []
    
    clean_student = [s.strip().lower() for s in student_skills if s and s.strip()]
    required = career_info.get("required_skills", [])
    
    matched = []
    missing = []
    
    for req in required:
        req_clean = req.lower()
        # check partial match or exact
        if any(req_clean in cs or cs in req_clean for cs in clean_student):
            matched.append(req)
        else:
            missing.append(req)
            
    match_percentage = int((len(matched) / len(required) * 100)) if required else 0
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": match_percentage
    }


def recommend_careers(
    aptitude_scores: Dict[str, float],
    riasec_scores: Dict[str, float],
    dominant_interest: Optional[str] = None,
    student_skills: Optional[List[str]] = None,
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Main recommendation function.
    Returns ranked list of top career matches with confidence percentage,
    career metadata, skill gap analysis, and tailored justification.
    """
    student_vec = calculate_student_vector(aptitude_scores, riasec_scores)
    
    # Build career vector matrix
    career_keys = list(CAREER_DATABASE.keys())
    career_vectors = np.array([CAREER_DATABASE[k]["vector"] for k in career_keys])
    
    # Compute Cosine Similarities
    similarities = cosine_similarity(student_vec, career_vectors)[0]
    
    scored_careers = []
    
    for idx, key in enumerate(career_keys):
        career = CAREER_DATABASE[key]
        raw_sim = float(similarities[idx])
        
        # Domain alignment bonus: if dominant_interest matches career category
        domain_bonus = 0.0
        if dominant_interest and dominant_interest in career["category"]:
            domain_bonus = 0.08
            
        final_score = min(0.98, raw_sim * 0.90 + domain_bonus)
        confidence_pct = int(round(final_score * 100))
        
        # Skill gap
        skill_gap = calculate_skill_gap(career, student_skills)
        
        # Match reason
        reasons = []
        if aptitude_scores.get("logical", 0) >= 7 and "Technology" in career["category"]:
            reasons.append("High analytical and logical problem-solving aptitude")
        if aptitude_scores.get("numerical", 0) >= 7 and ("Financial" in career["title"] or "Data" in career["title"]):
            reasons.append("Strong quantitative proficiency")
        if riasec_scores.get("A", 0) >= 6 and "Design" in career["category"]:
            reasons.append("Prominent artistic and visual creativity score")
        if riasec_scores.get("S", 0) >= 6 and "Healthcare" in career["category"]:
            reasons.append("Strong interpersonal empathy and social orientation")
        if riasec_scores.get("E", 0) >= 6 and ("Product" in career["title"] or "Marketing" in career["title"]):
            reasons.append("High enterprising and strategic leadership mindset")
            
        if not reasons:
            reasons.append(f"Strong overall alignment with your {dominant_interest or 'assessed'} traits")
            
        scored_careers.append({
            "id": key,
            "title": career["title"],
            "category": career["category"],
            "description": career["description"],
            "confidence": confidence_pct,
            "confidence_label": "Very Strong Fit" if confidence_pct >= 85 else ("Strong Match" if confidence_pct >= 75 else "Good Potential"),
            "reasons": reasons,
            "avg_salary_inr": career["avg_salary_inr"],
            "market_demand": career["market_demand"],
            "required_skills": career["required_skills"],
            "soft_skills": career["soft_skills"],
            "skill_gap": skill_gap,
            "roadmap": career["roadmap"],
            "resources": career["resources"]
        })
        
    # Sort descending by confidence
    scored_careers.sort(key=lambda x: x["confidence"], reverse=True)
    top_matches = scored_careers[:top_k]
    
    return {
        "primary_recommendation": top_matches[0] if top_matches else None,
        "recommendations": top_matches,
        "top_careers_list": [c["title"] for c in top_matches],
        "top_confidence": top_matches[0]["confidence"] if top_matches else 75
    }


def get_career_by_id(career_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full details of a specific career."""
    return CAREER_DATABASE.get(career_id)
