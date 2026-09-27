import json
from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

import ai

app = Flask(__name__)
app.config["SECRET_KEY"] = "career-guidance-super-secret-key-2026"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///career.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"


# ---------------------------------------------------------------------------
# DATABASE MODELS
# ---------------------------------------------------------------------------

class Student(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    class_name = db.Column(db.String(50), nullable=True)
    skills = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    test_results = db.relationship(
        "TestResult",
        backref="student",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="desc(TestResult.created_at)"
    )

    def get_skills_list(self):
        if not self.skills:
            return []
        return [s.strip() for s in self.skills.split(",") if s.strip()]


class TestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("student.id"), nullable=False)
    
    # Aptitude breakdown (0 - 10 scale)
    aptitude_score = db.Column(db.Integer, default=0)
    numerical_score = db.Column(db.Integer, default=0)
    logical_score = db.Column(db.Integer, default=0)
    verbal_score = db.Column(db.Integer, default=0)
    technical_score = db.Column(db.Integer, default=0)

    # Interest & RIASEC
    interest = db.Column(db.String(100), nullable=True)
    riasec_json = db.Column(db.Text, nullable=True)

    # Recommendations
    career = db.Column(db.String(150), nullable=True)
    confidence = db.Column(db.Integer, default=0)
    recommendations_json = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def get_riasec_dict(self):
        if not self.riasec_json:
            return {"R": 0, "I": 0, "A": 0, "S": 0, "E": 0, "C": 0}
        try:
            return json.loads(self.riasec_json)
        except Exception:
            return {"R": 0, "I": 0, "A": 0, "S": 0, "E": 0, "C": 0}

    def get_recommendations_list(self):
        if not self.recommendations_json:
            return []
        try:
            return json.loads(self.recommendations_json)
        except Exception:
            return []


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Student, int(user_id))


# ---------------------------------------------------------------------------
# QUESTION REPOSITORIES
# ---------------------------------------------------------------------------

APTITUDE_QUESTIONS = [
    # Numerical Aptitude (3 questions)
    {
        "id": 1,
        "category": "numerical",
        "category_name": "Numerical Ability",
        "question": "A laptop originally priced at $800 is offered with a 25% discount. What is its final sale price?",
        "options": ["$550", "$600", "$650", "$700"],
        "answer": "$600"
    },
    {
        "id": 2,
        "category": "numerical",
        "category_name": "Numerical Ability",
        "question": "If a vehicle travels 240 kilometers in 3 hours, what is its average speed?",
        "options": ["70 km/h", "75 km/h", "80 km/h", "85 km/h"],
        "answer": "80 km/h"
    },
    {
        "id": 3,
        "category": "numerical",
        "category_name": "Numerical Ability",
        "question": "What is the next number in the geometric series: 3, 9, 27, 81, ?",
        "options": ["162", "216", "243", "324"],
        "answer": "243"
    },
    # Logical Reasoning (3 questions)
    {
        "id": 4,
        "category": "logical",
        "category_name": "Logical Reasoning",
        "question": "Identify the number that continues the pattern: 2, 6, 12, 20, 30, ?",
        "options": ["38", "40", "42", "44"],
        "answer": "42"
    },
    {
        "id": 5,
        "category": "logical",
        "category_name": "Logical Reasoning",
        "question": "Premise: All software engineers solve problems. Maya solves problems. Which statement must logically follow?",
        "options": [
            "Maya is definitely a software engineer",
            "Maya might be a software engineer",
            "Maya cannot be an engineer",
            "Software engineers are the only problem solvers"
        ],
        "answer": "Maya might be a software engineer"
    },
    {
        "id": 6,
        "category": "logical",
        "category_name": "Logical Reasoning",
        "question": "Algorithm is to Software as Blueprint is to: ?",
        "options": ["Computer", "Building", "Code", "Contractor"],
        "answer": "Building"
    },
    # Verbal Comprehension (2 questions)
    {
        "id": 7,
        "category": "verbal",
        "category_name": "Verbal Ability",
        "question": "Choose the word most nearly OPPOSITE in meaning to 'INNOVATIVE':",
        "options": ["Creative", "Conventional", "Dynamic", "Pioneering"],
        "answer": "Conventional"
    },
    {
        "id": 8,
        "category": "verbal",
        "category_name": "Verbal Ability",
        "question": "Fill in the blank: Successful leaders provide _______ guidance that eliminates team ambiguity.",
        "options": ["lucid", "cryptic", "superfluous", "obscure"],
        "answer": "lucid"
    },
    # Technical & Computational Thinking (2 questions)
    {
        "id": 9,
        "category": "technical",
        "category_name": "Technical Thinking",
        "question": "In an unsorted list of 100 items, what is the maximum number of comparisons needed to find a specific element in the worst case?",
        "options": ["10", "50", "100", "1000"],
        "answer": "100"
    },
    {
        "id": 10,
        "category": "technical",
        "category_name": "Technical Thinking",
        "question": "If Condition A is True and Condition B is False, what does the Boolean expression 'not (A and B)' evaluate to?",
        "options": ["True", "False", "Error", "Undefined"],
        "answer": "True"
    }
]

INTEREST_QUESTIONS = [
    {
        "id": 1,
        "question": "When starting a major project, which task energizes you most?",
        "options": [
            {"text": "Building a working technical prototype or handling physical equipment", "code": "R"},
            {"text": "Investigating research data, analyzing hypotheses, and uncovering patterns", "code": "I"},
            {"text": "Designing the visual identity, UI aesthetics, or artistic media", "code": "A"},
            {"text": "Pitching the vision to stakeholders and organizing the leadership team", "code": "E"}
        ]
    },
    {
        "id": 2,
        "question": "During your free time, which activity do you find most fulfilling?",
        "options": [
            {"text": "Tinkering with gadgets, computer hardware, or hands-on DIY projects", "code": "R"},
            {"text": "Solving complex logic puzzles or exploring scientific discoveries", "code": "I"},
            {"text": "Creative writing, digital illustration, composing music, or video editing", "code": "A"},
            {"text": "Mentoring a peer, teaching a skill, or volunteering in community care", "code": "S"}
        ]
    },
    {
        "id": 3,
        "question": "Which workshop topic would you be most eager to attend?",
        "options": [
            {"text": "Applied Robotics and Automation Engineering", "code": "R"},
            {"text": "Advanced Machine Learning and Computational Statistics", "code": "I"},
            {"text": "Empathetic Communication and Psychological Wellbeing", "code": "S"},
            {"text": "Financial Portfolio Management and Corporate Budgeting", "code": "C"}
        ]
    },
    {
        "id": 4,
        "question": "When tackling an unfamiliar challenge, what is your initial instinct?",
        "options": [
            {"text": "Get straight to hands-on testing until something works practically", "code": "R"},
            {"text": "Deep-dive into documentation, theory, and empirical data", "code": "I"},
            {"text": "Brainstorm original, out-of-the-box creative concepts", "code": "A"},
            {"text": "Formulate a step-by-step checklist and operational timeline", "code": "C"}
        ]
    },
    {
        "id": 5,
        "question": "If you were founding a new organization, which department would you run?",
        "options": [
            {"text": "Product Development and Technical Architecture", "code": "R"},
            {"text": "Brand Design, Creative Direction, and User Experience", "code": "A"},
            {"text": "Business Strategy, Investor Relations, and Sales Growth", "code": "E"},
            {"text": "Compliance, Operations, and Financial Bookkeeping", "code": "C"}
        ]
    },
    {
        "id": 6,
        "question": "Which professional environment sounds most inspiring?",
        "options": [
            {"text": "An analytical research laboratory or data think-tank", "code": "I"},
            {"text": "A vibrant digital design studio with freedom for creative expression", "code": "A"},
            {"text": "A supportive counseling center, school, or healthcare clinic", "code": "S"},
            {"text": "A high-stakes corporate boardroom or entrepreneurial venture hub", "code": "E"}
        ]
    },
    {
        "id": 7,
        "question": "What kind of real-world problem motivates you the most?",
        "options": [
            {"text": "Developing resilient infrastructure and clean technology systems", "code": "R"},
            {"text": "Unraveling complex mathematical or biomedical mysteries", "code": "I"},
            {"text": "Addressing mental health, community wellness, and social empowerment", "code": "S"},
            {"text": "Negotiating trade deals and scaling global enterprises", "code": "E"}
        ]
    },
    {
        "id": 8,
        "question": "When working with numbers, what gives you the greatest satisfaction?",
        "options": [
            {"text": "Calibrating sensor telemetry and engineering dimensions", "code": "R"},
            {"text": "Building predictive mathematical and AI models", "code": "I"},
            {"text": "Analyzing market share, ROI, and customer lifetime value", "code": "E"},
            {"text": "Auditing balance sheets to guarantee 100% financial accuracy", "code": "C"}
        ]
    },
    {
        "id": 9,
        "question": "If contributing to a digital media publication, what would you choose?",
        "options": [
            {"text": "Directing motion graphics, layout visuals, and interactive styling", "code": "A"},
            {"text": "Authoring in-depth investigative reports based on extensive research", "code": "I"},
            {"text": "Hosting interviews, audience discussions, and community roundtables", "code": "S"},
            {"text": "Coordinating editorial calendars, cataloging archives, and content schedules", "code": "C"}
        ]
    },
    {
        "id": 10,
        "question": "Which core capability would you most like to elevate?",
        "options": [
            {"text": "Building full-stack cloud software and robust codebases", "code": "R"},
            {"text": "Mastering UI/UX wireframing, typography, and design aesthetics", "code": "A"},
            {"text": "Active listening, empathy, and therapeutic conflict resolution", "code": "S"},
            {"text": "Persuasive storytelling, public speaking, and negotiation", "code": "E"}
        ]
    },
    {
        "id": 11,
        "question": "Which core principle resonates most with your working style?",
        "options": [
            {"text": "Methodical precision, reliability, and structured order", "code": "C"},
            {"text": "Intellectual curiosity and uncovering empirical truth", "code": "I"},
            {"text": "Authentic self-expression, novelty, and aesthetic beauty", "code": "A"},
            {"text": "Compassion, human service, and uplifting others", "code": "S"}
        ]
    },
    {
        "id": 12,
        "question": "What completed deliverable would make you feel most accomplished?",
        "options": [
            {"text": "A deployed software tool or physical automated device", "code": "R"},
            {"text": "A comprehensive data analysis report answering a key hypothesis", "code": "I"},
            {"text": "A commercial venture that successfully turned a profit", "code": "E"},
            {"text": "An impeccably structured, error-free database and filing system", "code": "C"}
        ]
    }
]


# ---------------------------------------------------------------------------
# ROUTES: AUTHENTICATION & LANDING
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        age = request.form.get("age")
        class_name = request.form.get("class_name")
        skills = request.form.get("skills", "").strip()

        if not name or not email or not password:
            flash("Please fill in all required fields.", "danger")
            return redirect(url_for("register"))

        existing_student = Student.query.filter_by(email=email).first()
        if existing_student:
            flash("Email is already registered. Please log in.", "warning")
            return redirect(url_for("login"))

        hashed_password = generate_password_hash(password)
        student = Student(
            name=name,
            email=email,
            password=hashed_password,
            age=int(age) if age and age.isdigit() else None,
            class_name=class_name,
            skills=skills
        )

        db.session.add(student)
        db.session.commit()

        login_user(student)
        flash("Welcome to AI Career Guidance! Your account is created.", "success")
        return redirect(url_for("dashboard"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        student = Student.query.filter_by(email=email).first()
        if student and check_password_hash(student.password, password):
            login_user(student)
            flash("Login successful! Welcome back.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email or password. Please try again.", "danger")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been securely logged out.", "info")
    return redirect(url_for("home"))


# ---------------------------------------------------------------------------
# ROUTES: DASHBOARD & PROFILE
# ---------------------------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():
    latest_result = TestResult.query.filter_by(
        student_id=current_user.id
    ).order_by(TestResult.id.desc()).first()

    return render_template(
        "dashboard.html",
        student=current_user,
        result=latest_result
    )


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        current_user.name = request.form.get("name", current_user.name).strip()
        age = request.form.get("age")
        current_user.age = int(age) if age and age.isdigit() else current_user.age
        current_user.class_name = request.form.get("class_name", current_user.class_name)
        current_user.skills = request.form.get("skills", "").strip()

        db.session.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("dashboard"))

    return render_template("profile.html", student=current_user)


# ---------------------------------------------------------------------------
# ROUTES: APTITUDE ASSESSMENT
# ---------------------------------------------------------------------------

@app.route("/aptitude", methods=["GET", "POST"])
@login_required
def aptitude():
    if request.method == "POST":
        total_score = 0
        cat_scores = {
            "numerical": 0,
            "logical": 0,
            "verbal": 0,
            "technical": 0
        }

        for q in APTITUDE_QUESTIONS:
            qid = q["id"]
            user_ans = request.form.get(f"question_{qid}")
            if user_ans == q["answer"]:
                total_score += 1
                cat_scores[q["category"]] += 1

        # Normalize category scores to a 10-point scale for weighting
        numerical_norm = int(round((cat_scores["numerical"] / 3.0) * 10))
        logical_norm = int(round((cat_scores["logical"] / 3.0) * 10))
        verbal_norm = int(round((cat_scores["verbal"] / 2.0) * 10))
        technical_norm = int(round((cat_scores["technical"] / 2.0) * 10))

        # Check if student already has a pending test result without interest
        test_result = TestResult.query.filter_by(
            student_id=current_user.id
        ).order_by(TestResult.id.desc()).first()

        if not test_result or test_result.interest is not None:
            # Create a fresh record for this evaluation cycle
            test_result = TestResult(student_id=current_user.id)
            db.session.add(test_result)

        test_result.aptitude_score = total_score
        test_result.numerical_score = numerical_norm
        test_result.logical_score = logical_norm
        test_result.verbal_score = verbal_norm
        test_result.technical_score = technical_norm
        test_result.created_at = datetime.now(timezone.utc)

        db.session.commit()

        flash("Aptitude test completed successfully! Next, complete your interest assessment.", "success")
        return redirect(url_for("aptitude_result"))

    return render_template(
        "aptitude.html",
        questions=APTITUDE_QUESTIONS
    )


@app.route("/aptitude_result")
@login_required
def aptitude_result():
    result = TestResult.query.filter_by(
        student_id=current_user.id
    ).order_by(TestResult.id.desc()).first()

    if not result:
        flash("Please take the Aptitude Test first.", "warning")
        return redirect(url_for("aptitude"))

    return render_template(
        "aptitude_result.html",
        score=result.aptitude_score,
        total=len(APTITUDE_QUESTIONS),
        numerical=result.numerical_score,
        logical=result.logical_score,
        verbal=result.verbal_score,
        technical=result.technical_score
    )


# ---------------------------------------------------------------------------
# ROUTES: INTEREST ASSESSMENT (RIASEC)
# ---------------------------------------------------------------------------

@app.route("/interest", methods=["GET", "POST"])
@login_required
def interest():
    if request.method == "POST":
        riasec_counts = {"R": 0, "I": 0, "A": 0, "S": 0, "E": 0, "C": 0}

        for q in INTEREST_QUESTIONS:
            qid = q["id"]
            selected_code = request.form.get(f"question_{qid}")
            if selected_code and selected_code in riasec_counts:
                riasec_counts[selected_code] += 1

        # Normalize to 0-10 scale
        total_answers = sum(riasec_counts.values()) or 1
        riasec_scaled = {
            k: int(round((v / 6.0) * 10))
            for k, v in riasec_counts.items()
        }

        # Map top traits to dominant category
        # Holland code grouping:
        # High I / R -> Tech
        # High A -> Creative Arts
        # High S -> Healthcare & Social
        # High E / C -> Business & Management
        category_weights = {
            "Technology & Computer Science": riasec_counts["I"] * 1.5 + riasec_counts["R"] * 1.2,
            "Design & Creative Arts": riasec_counts["A"] * 2.0 + riasec_counts["I"] * 0.5,
            "Healthcare & Social Services": riasec_counts["S"] * 2.0 + riasec_counts["I"] * 0.8,
            "Business & Management": riasec_counts["E"] * 1.8 + riasec_counts["C"] * 1.2
        }

        dominant_category = max(category_weights, key=category_weights.get)

        # Retrieve student's latest test result
        test_result = TestResult.query.filter_by(
            student_id=current_user.id
        ).order_by(TestResult.id.desc()).first()

        if not test_result:
            test_result = TestResult(
                student_id=current_user.id,
                aptitude_score=7,
                numerical_score=7,
                logical_score=7,
                verbal_score=7,
                technical_score=7
            )
            db.session.add(test_result)

        # Update interest details in database
        test_result.interest = dominant_category
        test_result.riasec_json = json.dumps(riasec_scaled)

        # Generate AI recommendations
        aptitude_dict = {
            "numerical": test_result.numerical_score or 6,
            "logical": test_result.logical_score or 6,
            "verbal": test_result.verbal_score or 6,
            "technical": test_result.technical_score or 6
        }

        student_skills = current_user.get_skills_list()

        ai_output = ai.recommend_careers(
            aptitude_scores=aptitude_dict,
            riasec_scores=riasec_scaled,
            dominant_interest=dominant_category,
            student_skills=student_skills,
            top_k=4
        )

        test_result.career = ai_output["primary_recommendation"]["title"]
        test_result.confidence = ai_output["primary_recommendation"]["confidence"]
        test_result.recommendations_json = json.dumps(ai_output["recommendations"])

        db.session.commit()

        flash("Interest assessment submitted! Your career profile is ready.", "success")
        return redirect(url_for("interest_result"))

    return render_template(
        "interest.html",
        questions=INTEREST_QUESTIONS
    )


@app.route("/interest_result")
@login_required
def interest_result():
    result = TestResult.query.filter_by(
        student_id=current_user.id
    ).order_by(TestResult.id.desc()).first()

    if not result or not result.interest:
        flash("Please complete the Interest Test first.", "warning")
        return redirect(url_for("interest"))

    riasec_data = result.get_riasec_dict()

    return render_template(
        "interest_result.html",
        result=result.interest,
        riasec=riasec_data,
        riasec_names=ai.RIASEC_NAMES
    )


# ---------------------------------------------------------------------------
# ROUTES: AI RECOMMENDATION & CAREER ROADMAP
# ---------------------------------------------------------------------------

@app.route("/recommendation")
@login_required
def recommendation():
    result = TestResult.query.filter_by(
        student_id=current_user.id
    ).order_by(TestResult.id.desc()).first()

    if not result:
        flash("Please complete the Aptitude and Interest tests to get AI recommendations.", "warning")
        return redirect(url_for("dashboard"))

    if not result.interest:
        flash("Please complete the Interest Test first.", "info")
        return redirect(url_for("interest"))

    # If recommendations not yet generated or outdated, recalculate
    recommendations = result.get_recommendations_list()
    if not recommendations:
        aptitude_dict = {
            "numerical": result.numerical_score or 6,
            "logical": result.logical_score or 6,
            "verbal": result.verbal_score or 6,
            "technical": result.technical_score or 6
        }
        ai_output = ai.recommend_careers(
            aptitude_scores=aptitude_dict,
            riasec_scores=result.get_riasec_dict(),
            dominant_interest=result.interest,
            student_skills=current_user.get_skills_list(),
            top_k=4
        )
        recommendations = ai_output["recommendations"]
        result.career = ai_output["primary_recommendation"]["title"]
        result.confidence = ai_output["primary_recommendation"]["confidence"]
        result.recommendations_json = json.dumps(recommendations)
        db.session.commit()

    return render_template(
        "recommendation.html",
        recommendations=recommendations,
        top_career=result.career,
        confidence=result.confidence,
        score=result.aptitude_score,
        interest=result.interest,
        student=current_user,
        result=result
    )


@app.route("/career/<career_id>")
@login_required
def career_detail(career_id):
    career_info = ai.get_career_by_id(career_id)
    if not career_info:
        flash("Career path not found.", "warning")
        return redirect(url_for("recommendation"))

    skill_gap = ai.calculate_skill_gap(career_info, current_user.get_skills_list())
    return render_template(
        "career_detail.html",
        career=career_info,
        skill_gap=skill_gap,
        student=current_user
    )


@app.route("/report")
@login_required
def report():
    result = TestResult.query.filter_by(
        student_id=current_user.id
    ).order_by(TestResult.id.desc()).first()

    if not result or not result.career:
        flash("Please complete your assessments before generating a report.", "warning")
        return redirect(url_for("dashboard"))

    recommendations = result.get_recommendations_list()
    primary = recommendations[0] if recommendations else None

    return render_template(
        "report.html",
        student=current_user,
        result=result,
        primary=primary,
        recommendations=recommendations,
        riasec=result.get_riasec_dict(),
        riasec_names=ai.RIASEC_NAMES
    )


# ---------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------

with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True, port=5000)