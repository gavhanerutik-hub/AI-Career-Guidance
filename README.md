# 🎓 AI Career Guidance Platform

An intelligent, data-driven career counseling and guidance platform built with Python, Flask, and Scikit-Learn. The system assesses cognitive aptitude across multiple domains, evaluates vocational interests using the **Holland RIASEC** model, and employs **machine learning cosine similarity matching** to recommend tailored careers complete with skill gap analysis, compensation insights, and milestone-by-milestone 4-phase learning roadmaps.

**Author:** Rutik Gavhane

---

## 🌟 Key Features

### 1. 🧠 Multi-Domain Cognitive Aptitude Assessment
- Evaluates 10 structured challenges across 4 essential cognitive dimensions:
  - **Numerical Ability** (percentages, rates, geometric progressions)
  - **Logical Reasoning** (pattern series, deductive reasoning, analogies)
  - **Verbal Comprehension** (contextual vocabulary, sentence logic)
  - **Technical & Computational Thinking** (worst-case search complexity, Boolean algebra)
- Provides immediate visual domain breakdown progress bars.

### 2. ❤️ Holland RIASEC Vocational Assessment
- 12 real-world situational scenarios mapping student instincts to the proven Holland Occupational Themes:
  - **R** – Realistic (Hands-on, mechanical, physical systems)
  - **I** – Investigative (Analytical, research, scientific problem-solving)
  - **A** – Artistic (Creative expression, design, visual storytelling)
  - **S** – Social (Empathetic counseling, education, community service)
  - **E** – Enterprising (Leadership, business strategy, negotiation)
  - **C** – Conventional (Structured organization, precision, compliance)

### 3. 🤖 Machine Learning Recommendation Engine (`ai/career_engine.py`)
- Powered by `scikit-learn` vector normalization and multi-dimensional cosine similarity.
- Compares normalized 10-dimensional candidate profiles against a curated knowledge base of 20+ specialized high-growth careers.
- Generates ranked career matches with confidence percentages (e.g., 96% Match), domain alignment tags, and personalized explanations ("Why this career fits you").

### 4. 🎯 Skill Gap Analysis & 4-Phase Roadmaps
- **Skill Gap Analysis**: Compares candidate's declared skills against industry requirements to highlight:
  - ✅ Skills already acquired
  - 🎯 Priority skills to learn next
  - 💡 Essential soft skills
- **4-Phase Learning Roadmaps**: Actionable pathways from beginner foundations through intermediate tools, advanced systems, and portfolio/certification readiness.
- **Curated Free Resources**: Direct links to top learning platforms (roadmap.sh, Harvard CS50, DeepLearning.AI, freeCodeCamp, etc.).

### 5. 📊 Interactive Dashboard & Printable PDF Reports
- **Interactive Analytics**: Powered by **Chart.js**, rendering aptitude bar charts and RIASEC interest radar diagrams.
- **Official Career Report (`/report`)**: Clean, formatted print layout (`@media print`) ready to export as a PDF or physical printout.
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
│   ├── aptitude.html                # 10-question multi-domain aptitude test
│   ├── aptitude_result.html         # Aptitude score & category breakdown
│   ├── interest.html                # 12-question Holland RIASEC assessment
│   ├── interest_result.html         # RIASEC profile & dominant field
│   ├── recommendation.html          # Ranked AI matches, skill gaps, roadmaps
│   ├── career_detail.html           # Dedicated single-career deep-dive
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
