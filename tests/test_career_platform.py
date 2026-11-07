"""
tests/test_career_platform.py
Automated test suite for AI Career Guidance Platform:
- Authentication & User Management
- Assessment Scoring (Aptitude & RIASEC)
- AI / ML Recommendation Engine
- End-to-End User Journey (Aptitude -> Interest -> Recommendation -> Report)
"""

import unittest
import json
from app import app, db, Student, TestResult, APTITUDE_QUESTIONS, INTEREST_QUESTIONS
import ai


class CareerPlatformTestCase(unittest.TestCase):

    def setUp(self):
        """Configure test client and in-memory test database."""
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

        with app.app_context():
            db.create_all()

    def tearDown(self):
        """Clean up database after each test."""
        with app.app_context():
            db.session.remove()
            db.drop_all()

    # -----------------------------------------------------------------------
    # 1. AI Recommendation Engine Tests
    # -----------------------------------------------------------------------

    def test_ai_engine_recommendation_schema(self):
        """Verify the AI engine returns expected schema and keys."""
        aptitude = {"numerical": 8, "logical": 9, "verbal": 7, "technical": 9}
        riasec = {"R": 7, "I": 9, "A": 3, "S": 2, "E": 4, "C": 6}
        result = ai.recommend_careers(
            aptitude_scores=aptitude,
            riasec_scores=riasec,
            dominant_interest="Technology & Computer Science",
            student_skills=["Python", "SQL"]
        )

        self.assertIn("primary_recommendation", result)
        self.assertIn("recommendations", result)
        self.assertIn("top_careers_list", result)
        self.assertGreaterEqual(len(result["recommendations"]), 3)

        primary = result["primary_recommendation"]
        self.assertIn("title", primary)
        self.assertIn("category", primary)
        self.assertIn("confidence", primary)
        self.assertIn("roadmap", primary)
        self.assertIn("resources", primary)
        self.assertEqual(len(primary["roadmap"]), 4)

    def test_ai_engine_skill_gap_analysis(self):
        """Verify skill gap accurately detects possessed and missing skills."""
        career_info = ai.get_career_by_id("software_engineer")
        self.assertIsNotNone(career_info)

        gap = ai.calculate_skill_gap(career_info, ["Python", "Git"])
        self.assertIn("Python", gap["matched_skills"])
        self.assertIn("Git", gap["matched_skills"])
        self.assertIn("Data Structures", gap["missing_skills"])
        self.assertGreater(gap["match_percentage"], 0)

    def test_ai_engine_creative_arts_matching(self):
        """Verify students with high Artistic score match Creative Arts careers."""
        aptitude = {"numerical": 4, "logical": 6, "verbal": 8, "technical": 5}
        riasec = {"R": 2, "I": 4, "A": 10, "S": 6, "E": 5, "C": 3}
        result = ai.recommend_careers(
            aptitude_scores=aptitude,
            riasec_scores=riasec,
            dominant_interest="Design & Creative Arts"
        )

        primary_title = result["primary_recommendation"]["title"]
        self.assertTrue(
            "Designer" in primary_title or "Artist" in primary_title or "Media" in primary_title,
            f"Expected design career, got {primary_title}"
        )

    # -----------------------------------------------------------------------
    # 2. Authentication & User Management Tests
    # -----------------------------------------------------------------------

    def test_user_registration_and_login(self):
        """Test student registration, password hashing, and login."""
        # Register new student
        response = self.client.post("/register", data={
            "name": "Arjun Patel",
            "email": "arjun@example.com",
            "password": "SecretPassword123",
            "age": "20",
            "class_name": "B.Tech / B.E. (Engineering)",
            "skills": "Python, SQL, Linux"
        }, follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome, Arjun Patel", response.data)

        # Verify in database
        with app.app_context():
            student = Student.query.filter_by(email="arjun@example.com").first()
            self.assertIsNotNone(student)
            self.assertEqual(student.name, "Arjun Patel")
            self.assertNotEqual(student.password, "SecretPassword123")  # Hashed
            self.assertEqual(student.get_skills_list(), ["Python", "SQL", "Linux"])

        # Logout
        logout_resp = self.client.get("/logout", follow_redirects=True)
        self.assertEqual(logout_resp.status_code, 200)

        # Login again
        login_resp = self.client.post("/login", data={
            "email": "arjun@example.com",
            "password": "SecretPassword123"
        }, follow_redirects=True)

        self.assertEqual(login_resp.status_code, 200)
        self.assertIn(b"Login successful", login_resp.data)

    def test_duplicate_registration_prevented(self):
        """Test that registering duplicate emails is rejected."""
        self.client.post("/register", data={
            "name": "User One",
            "email": "duplicate@example.com",
            "password": "Password123"
        })

        # Log out first user so register endpoint accepts new registration submission
        self.client.get("/logout")

        dup_response = self.client.post("/register", data={
            "name": "User Two",
            "email": "duplicate@example.com",
            "password": "Password456"
        }, follow_redirects=True)

        self.assertIn(b"Email is already registered", dup_response.data)

    # -----------------------------------------------------------------------
    # 3. End-to-End Assessment & Recommendation Journey
    # -----------------------------------------------------------------------

    def test_full_career_guidance_flow(self):
        """
        Verify end-to-end user journey:
        1. Register and log in.
        2. Submit Aptitude Test -> Check score saved in DB.
        3. Submit Interest Test -> Check RIASEC and AI recommendations generated.
        4. Access Recommendation Page -> Verify top matches rendered.
        5. Access Career Detail Page -> Verify 200 OK.
        6. Access Printable Report -> Verify 200 OK.
        """
        # Step 1: Register
        self.client.post("/register", data={
            "name": "Neha Gupta",
            "email": "neha@example.com",
            "password": "TestPassword2026",
            "class_name": "B.Sc / BCA (Computer/Science)",
            "skills": "Python, Machine Learning"
        }, follow_redirects=True)

        # Step 2: Submit Aptitude Test (Submit correct answers for all 10 questions)
        aptitude_form_data = {}
        for q in APTITUDE_QUESTIONS:
            aptitude_form_data[f"question_{q['id']}"] = q["answer"]

        apt_resp = self.client.post("/aptitude", data=aptitude_form_data, follow_redirects=True)
        self.assertEqual(apt_resp.status_code, 200)
        self.assertIn(b"Your Aptitude Results", apt_resp.data)
        self.assertIn(b"10", apt_resp.data)  # Perfect score

        # Verify database record
        with app.app_context():
            student = Student.query.filter_by(email="neha@example.com").first()
            result = TestResult.query.filter_by(student_id=student.id).first()
            self.assertIsNotNone(result)
            self.assertEqual(result.aptitude_score, 10)
            self.assertEqual(result.numerical_score, 10)
            self.assertEqual(result.logical_score, 10)
            self.assertEqual(result.verbal_score, 10)
            self.assertEqual(result.technical_score, 10)

        # Step 3: Submit Interest Assessment (Select code 'I' and 'R' mostly)
        interest_form_data = {}
        for q in INTEREST_QUESTIONS:
            interest_form_data[f"question_{q['id']}"] = q["options"][0]["code"]

        int_resp = self.client.post("/interest", data=interest_form_data, follow_redirects=True)
        self.assertEqual(int_resp.status_code, 200)
        self.assertIn(b"Your Holland Interest Profile", int_resp.data)

        # Verify interest saved and recommendation generated
        with app.app_context():
            result = TestResult.query.filter_by(student_id=student.id).first()
            self.assertIsNotNone(result.interest)
            self.assertIsNotNone(result.career)
            self.assertGreater(result.confidence, 60)
            self.assertGreater(len(result.get_recommendations_list()), 0)

        # Step 4: Visit Recommendation page
        recom_resp = self.client.get("/recommendation")
        self.assertEqual(recom_resp.status_code, 200)
        self.assertIn(b"Your Personalized Career Matches", recom_resp.data)
        self.assertIn(b"Skill Gap Analysis", recom_resp.data)
        self.assertIn(b"Step-by-Step Learning Roadmap", recom_resp.data)

        # Step 5: Visit Career Detail page
        detail_resp = self.client.get("/career/software_engineer")
        self.assertEqual(detail_resp.status_code, 200)
        self.assertIn(b"Software Engineer", detail_resp.data)

        # Step 6: Visit Report page
        report_resp = self.client.get("/report")
        self.assertEqual(report_resp.status_code, 200)
        self.assertIn(b"Official Career Assessment Report", report_resp.data)
        self.assertIn(b"Neha Gupta", report_resp.data)

    # -----------------------------------------------------------------------
    # 4. Expanded Features Tests: Holland Code, Catalog & Skill API
    # -----------------------------------------------------------------------

    def test_holland_profile_generation(self):
        """Verify Holland 3-letter code and archetype persona generation."""
        riasec_scores = {"R": 8, "I": 9, "A": 3, "S": 2, "E": 4, "C": 7}
        profile = ai.calculate_holland_profile(riasec_scores)

        self.assertIn("code", profile)
        self.assertEqual(len(profile["code"]), 3)
        self.assertEqual(profile["code"][:2], "IR")
        self.assertIn("persona", profile)
        self.assertIn("title", profile["persona"])
        self.assertIn("tagline", profile["persona"])
        self.assertIn("motivators", profile["persona"])

    def test_careers_catalog_page_and_search(self):
        """Verify careers directory renders and filters correctly."""
        # Test default catalog view
        resp = self.client.get("/careers")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Explore High-Growth", resp.data)
        self.assertIn(b"Software Engineer", resp.data)

        # Test search query
        search_resp = self.client.get("/careers?q=Python")
        self.assertEqual(search_resp.status_code, 200)
        self.assertIn(b"Software Engineer", search_resp.data)

        # Test category filter
        cat_resp = self.client.get("/careers?category=Design%20%26%20Creative%20Arts")
        self.assertEqual(cat_resp.status_code, 200)
        self.assertIn(b"UI/UX Designer", cat_resp.data)

    def test_api_add_skill_endpoint(self):
        """Verify the interactive 1-click skill addition API."""
        # Register and log in
        self.client.post("/register", data={
            "name": "Kavya Sharma",
            "email": "kavya@example.com",
            "password": "Password123"
        })

        # Add skill via API
        resp = self.client.post(
            "/api/add_skill",
            data=json.dumps({"skill": "Docker"}),
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("Docker", data["skills"])

        # Verify persistence in database
        with app.app_context():
            student = Student.query.filter_by(email="kavya@example.com").first()
            self.assertIn("Docker", student.get_skills_list())

    def test_knowledge_base_expansion_depth(self):
        """Verify the career engine contains 20+ specialized high-growth careers."""
        all_careers = ai.get_all_careers()
        self.assertGreaterEqual(len(all_careers), 20)

        for career in all_careers:
            self.assertIn("title", career)
            self.assertIn("category", career)
            self.assertIn("avg_salary_inr", career)
            self.assertIn("market_demand", career)
            self.assertGreaterEqual(len(career.get("required_skills", [])), 3)
            self.assertEqual(len(career.get("roadmap", [])), 4)


if __name__ == "__main__":
    unittest.main()
