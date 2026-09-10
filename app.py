from flask import Flask, render_template, request, redirect, url_for, flash
from importlib import import_module
# Load the recommendation module without requiring the editor to resolve the
# project package statically.
recommend_career = import_module("ai.careet_recommendation").recommend_career
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

app = Flask(__name__)

app.config["SECRET_KEY"] = "career-guidance-secret-key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///career.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class Student(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=False)

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    age = db.Column(db.Integer)

    class_name = db.Column(db.String(30))


class TestResult(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    student_id = db.Column(
        db.Integer,
        db.ForeignKey("student.id"),
        nullable=False
    )

    aptitude_score = db.Column(db.Integer, default=0)

    interest = db.Column(db.String(100))

    career = db.Column(db.String(100))


@login_manager.user_loader
def load_user(user_id):

    return db.session.get(Student, int(user_id))


@app.route("/")
def home():

    return render_template("index.html")



@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        age = request.form.get("age")
        class_name = request.form.get("class_name")

        existing_student = Student.query.filter_by(
            email=email
        ).first()

        if existing_student:

            flash("Email already registered.")

            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        student = Student(
            name=name,
            email=email,
            password=hashed_password,
            age=int(age) if age else None,
            class_name=class_name
        )

        db.session.add(student)
        db.session.commit()

        flash("Registration successful! Please login.")

        return redirect(url_for("login"))

    return render_template("register.html")



@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        student = Student.query.filter_by(
            email=email
        ).first()

        if student and check_password_hash(
            student.password,
            password
        ):

            login_user(student)

            return redirect(url_for("dashboard"))

        flash("Invalid email or password.")

    return render_template("login.html")



@app.route("/dashboard")
@login_required
def dashboard():

    return render_template(
        "dashboard.html",
        student=current_user
    )


@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(url_for("home"))


with app.app_context():

    db.create_all()


@app.route("/aptitude", methods=["GET", "POST"])
@login_required
def aptitude():

    questions = [
        {
            "question": "What is 15 + 25?",
            "options": ["30", "35", "40", "45"],
            "answer": "40"
        },
        {
            "question": "What comes next: 2, 4, 6, 8, ?",
            "options": ["9", "10", "11", "12"],
            "answer": "10"
        },
        {
            "question": "Which number is prime?",
            "options": ["4", "6", "7", "9"],
            "answer": "7"
        },
        {
            "question": "If 5 × 6 = ?",
            "options": ["25", "30", "35", "40"],
            "answer": "30"
        },
        {
            "question": "Which is the largest?",
            "options": ["12", "21", "18", "15"],
            "answer": "21"
        },
        {
            "question": "100 ÷ 10 = ?",
            "options": ["5", "10", "15", "20"],
            "answer": "10"
        },
        {
            "question": "What is 9²?",
            "options": ["18", "72", "81", "90"],
            "answer": "81"
        },
        {
            "question": "Which comes next: A, C, E, G, ?",
            "options": ["H", "I", "J", "K"],
            "answer": "I"
        },
        {
            "question": "20% of 100 is?",
            "options": ["10", "20", "30", "40"],
            "answer": "20"
        },
        {
            "question": "Which one is different?",
            "options": ["Apple", "Mango", "Carrot", "Banana"],
            "answer": "Carrot"
        }
    ]

    if request.method == "POST":

        score = 0

        for i, question in enumerate(questions):

            user_answer = request.form.get(
                f"question_{i}"
            )

            if user_answer == question["answer"]:
                score += 1

        result = TestResult(
            student_id=current_user.id,
            aptitude_score=score
        )

        db.session.add(result)
        db.session.commit()

        return render_template(
            "aptitude_result.html",
            score=score,
            total=len(questions)
        )

    return render_template(
        "aptitude.html",
        questions=questions
    )


@app.route("/interest", methods=["GET", "POST"])
@login_required
def interest():

    questions = [

        {
            "id": 1,
            "question": "What do you enjoy doing most?",
            "options": [
                "Solving problems",
                "Drawing and designing",
                "Helping people",
                "Business and money"
            ]
        },

        {
            "id": 2,
            "question": "Which activity interests you?",
            "options": [
                "Programming",
                "Creative design",
                "Teaching",
                "Managing projects"
            ]
        },

        {
            "id": 3,
            "question": "Which subject do you prefer?",
            "options": [
                "Mathematics",
                "Arts",
                "Biology",
                "Economics"
            ]
        },

        {
            "id": 4,
            "question": "What type of work do you like?",
            "options": [
                "Working with computers",
                "Creating new things",
                "Working with people",
                "Leading a team"
            ]
        },

        {
            "id": 5,
            "question": "Which skill would you like to improve?",
            "options": [
                "Coding",
                "Creativity",
                "Communication",
                "Leadership"
            ]
        }

    ]

    if request.method == "POST":

        answers = []

        for question in questions:

            answer = request.form.get(
                f"question_{question['id']}"
            )

            if answer:
                answers.append(answer)


        # Simple interest classification

        tech_words = [
            "Programming",
            "Mathematics",
            "computers",
            "Coding",
            "Solving problems"
        ]

        creative_words = [
            "Drawing and designing",
            "Creative design",
            "Arts",
            "Creating new things",
            "Creativity"
        ]

        people_words = [
            "Helping people",
            "Teaching",
            "Working with people",
            "Communication"
        ]

        business_words = [
            "Business and money",
            "Managing projects",
            "Leading a team",
            "Leadership"
        ]


        tech_score = sum(
            answer in tech_words
            for answer in answers
        )

        creative_score = sum(
            answer in creative_words
            for answer in answers
        )

        people_score = sum(
            answer in people_words
            for answer in answers
        )

        business_score = sum(
            answer in business_words
            for answer in answers
        )


        scores = {
            "Technology & Computer Science": tech_score,
            "Design & Creative Arts": creative_score,
            "Healthcare & Social Services": people_score,
            "Business & Management": business_score
        }


        result = max(
            scores,
            key=scores.get
        )
        


        return render_template(
            "interest_result.html",
            result=result
        )


    return render_template(
        "interest.html",
        questions=questions
    )


def recommend_career(aptitude_score, interest):
    """Recommend careers based on aptitude score and interest."""
    
    careers_by_interest = {
        "Technology & Computer Science": {
            "careers": ["Software Developer", "Data Scientist", "AI Engineer", "Web Developer"],
            "confidence": 0.95 if aptitude_score >= 7 else 0.75
        },
        "Design & Creative Arts": {
            "careers": ["Graphic Designer", "UX/UI Designer", "Animator", "Artist"],
            "confidence": 0.90 if aptitude_score >= 6 else 0.70
        },
        "Healthcare & Social Services": {
            "careers": ["Doctor", "Nurse", "Social Worker", "Counselor"],
            "confidence": 0.92 if aptitude_score >= 7 else 0.72
        },
        "Business & Management": {
            "careers": ["Manager", "Entrepreneur", "Business Analyst", "Consultant"],
            "confidence": 0.88 if aptitude_score >= 6 else 0.68
        }
    }
    
    return careers_by_interest.get(
        interest,
        {
            "careers": ["General Career Path"],
            "confidence": 0.50
        }
    )


@app.route("/recommendation")
@login_required
def recommendation():

    result = TestResult.query.filter_by(
        student_id=current_user.id
    ).order_by(
        TestResult.id.desc()
    ).first()

    if not result:

        flash("Please complete the tests first.")

        return redirect(url_for("dashboard"))

    aptitude_score = result.aptitude_score

    interest = result.interest

    if not interest:

        flash("Please complete the Interest Test first.")

        return redirect(url_for("interest"))

    ai_result = recommend_career(
        aptitude_score,
        interest
    )

    result.career = ai_result["careers"][0]

    db.session.commit()

    return render_template(
        "recommendation.html",
        careers=ai_result["careers"],
        confidence=ai_result["confidence"],
        score=aptitude_score,
        interest=interest
    )
with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)