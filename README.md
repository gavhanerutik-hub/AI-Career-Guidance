# 🎓 AI Career Guidance Platform

An intelligent, data-driven career counseling and guidance platform built with Python, Flask, and Scikit-Learn. The system assesses cognitive aptitude across multiple domains, evaluates vocational interests using the **Holland RIASEC** model, and employs **machine learning cosine similarity matching** to recommend tailored careers complete with skill gap analysis, compensation insights, and milestone-by-milestone 4-phase learning roadmaps.

**Author:** Rutik Gavhane

---

## 🌟 Key Features

### 1. 🧠 Multi-Domain Cognitive Aptitude Assessment & Step-by-Step Learning
- Evaluates 10 structured challenges across 4 essential cognitive dimensions:
  - **Numerical Ability** (percentages, rates, geometric progressions)
  - **Logical Reasoning** (pattern series, deductive reasoning, analogies)
  - **Verbal Comprehension** (contextual vocabulary, sentence logic)
  - **Technical & Computational Thinking** (worst-case search complexity, Boolean algebra)
- **Live Progress Meter**: Interactive real-time completion tracker as questions are answered.
- **Detailed Step-by-Step Solutions Review**: Expandable accordion breaking down correct answers, candidate choices, and full pedagogical explanations for continuous learning.
- Dynamic calibrated domain progress bars with color-coded proficiency tiers.

### 2. ❤️ Holland RIASEC Vocational Profiling & Archetype Personas
- 12 real-world situational scenarios mapping student instincts to the proven Holland Occupational Themes:
  - **R** – Realistic (Hands-on, mechanical, physical systems)
  - **I** – Investigative (Analytical, research, scientific problem-solving)
  - **A** – Artistic (Creative expression, design, visual storytelling)
  - **S** – Social (Empathetic counseling, education, community service)
  - **E** – Enterprising (Leadership, business strategy, negotiation)
  - **C** – Conventional (Structured organization, precision, compliance)
- **Official 3-Letter Holland Code**: Generates standard vocational codes (e.g., `IRC`, `AES`, `ERC`).
- **Vocational Archetype Persona**: Assigns students intuitive personas (e.g., *The Technical Innovator & Systems Architect*, *The Creative Researcher*, *The Enterprise Strategist*) with driving motivators and ideal workplace environments.
- **Interactive Hexagon Radar Chart**: Embedded Chart.js polygon visualizing the candidate's complete 6-dimensional career passion envelope.

### 3. 🤖 Machine Learning Recommendation Engine (`ai/career_engine.py`)
- Powered by `scikit-learn` vector normalization and multi-dimensional cosine similarity.
- Compares normalized 10-dimensional candidate profiles against an expanded knowledge base of **22+ specialized high-growth careers** across 5 tracks:
  - Technology & Computer Science (Software Engineering, AI/ML, Data Science, Cybersecurity, Cloud/DevOps, Blockchain/Web3)
  - Design & Creative Arts (UI/UX Design, Game Design, Product Design, 3D Motion VFX)
  - Business & Management (Product Management, Financial Analysis, Quantitative Trading, Management Consulting, Digital Marketing, HR/Talent Strategy)
  - Engineering & Emerging Tech (Robotics & Automation, Renewable CleanTech)
  - Healthcare & Human Services (Clinical Psychology, Biomedical Engineering, Public Health, Bioinformatics)
- Generates ranked career matches with confidence percentages (e.g., 96% Match), domain alignment tags, and personalized explanations ("Why this career fits you").

### 4. 📚 Career Exploration Directory (`/careers`)
- Searchable catalog of all 22+ future-ready careers with keyword filtering (titles, required skills, descriptions).
- Category pill filters allowing students to easily explore cross-domain possibilities.
- Live skill match percentage badges for authenticated students.

### 5. 🎯 Interactive Skill Gap Analysis & 1-Click Profile Updates
- **Interactive Skill Gap**: Compares candidate's declared skills against industry requirements:
  - ✅ Skills already acquired
  - 🎯 Priority skills to learn next with **1-click "+ Add to Profile"** instant AJAX updates
  - 💡 Essential soft skills
- **4-Phase Learning Roadmaps**: Actionable pathways from beginner foundations through intermediate tools, advanced systems, and portfolio/certification readiness.
- **Curated Free Resources**: Direct links to top learning platforms (roadmap.sh, Harvard CS50, DeepLearning.AI, freeCodeCamp, MIT OCW, etc.).

### 6. 📊 Interactive Dashboard & Printable PDF Reports
- **Interactive Analytics**: Powered by **Chart.js**, rendering aptitude bar charts and RIASEC interest radar diagrams.
- **Official Career Report (`/report`)**: Clean, formatted print layout (`@media print`) ready to export as a PDF or physical printout with Holland Code and cognitive diagnostics.
- **Full Authentication & Profile Management**: Secure password hashing with Werkzeug, persistent SQLite storage via Flask-SQLAlchemy, and dynamic skill updates.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | Python 3, Flask 3.1 |
| **Authentication & ORM** | Flask-Login, Flask-SQLAlchemy, Werkzeug Security |
| **Database** | SQLite 3 |
| **AI / Machine Learning** | Scikit-Learn (`cosine_similarity`, `normalize`), NumPy |
| **Frontend & UI** | Semantic HTML5, Custom CSS3 Design System, Jinja2 Templates |
| **Data Visualizations** | Chart.js 4.4 |
| **Testing** | Python `unittest` |

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Clone and Navigate
```bash
git clone https://github.com/gavhanerutik-hub/AI-Career-Guidance.git
cd AI-Career-Guidance
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🧪 Running Automated Tests

Run the comprehensive unit and integration test suite:
```bash
python -m unittest tests/test_career_platform.py -v
```

All 6 test suites verify:
- AI recommendation schema, match confidence, and multi-factor vector math
- Creative arts, STEM, and business career alignment
- User registration, password hashing, and login/logout sessions
- Duplicate registration prevention
- End-to-end user journey (Register → Aptitude Assessment → Interest Assessment → AI Matches → Career Detail → PDF Report)

---

## 📁 Project Structure

```
AI-Career-Guidance/
│
├── ai/                              # AI & Machine Learning Package
│   ├── __init__.py                  # Package exports
│   └── career_engine.py             # RIASEC vector similarity, skill gap, knowledge base
│
├── static/                          # Static Assets
│   └── css/
│       └── style.css                # Complete modern responsive design system
│
├── templates/                       # Jinja2 HTML Templates
│   ├── base.html                    # Base layout with navbar, alerts, Chart.js CDN
│   ├── index.html                   # Landing page
│   ├── login.html                   # Student login
│   ├── register.html                # Student registration
│   ├── dashboard.html               # Analytics dashboard with Chart.js
│   ├── profile.html                 # Student profile & skill management
│   ├── careers.html                 # Searchable directory of 22+ careers with filters
│   ├── aptitude.html                # 10-question multi-domain aptitude test with live tracker
│   ├── aptitude_result.html         # Aptitude score, breakdown & step-by-step solutions
│   ├── interest.html                # 12-question Holland RIASEC assessment with live tracker
│   ├── interest_result.html         # RIASEC radar chart, 3-letter code & archetype persona
│   ├── recommendation.html          # Ranked AI matches, skill gaps, roadmaps
│   ├── career_detail.html           # Dedicated career deep-dive with 1-click skill addition
│   └── report.html                  # Official printable PDF report
│
├── tests/                           # Automated Test Suite
│   └── test_career_platform.py      # Unit & end-to-end integration tests
│
├── instance/                        # SQLite Local Database
│   └── career.db
│
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Project dependencies
├── README.md                        # Documentation
└── app.py                           # Main Flask application & routes
```

---

## 📄 License
This project is open-source and available for educational and career-guidance purposes.
