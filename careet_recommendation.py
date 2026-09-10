def recommend_career(aptitude_score, interest):

    careers = {

        "Technology & Computer Science": [
            "Software Developer",
            "Data Scientist",
            "AI/ML Engineer"
        ],

        "Design & Creative Arts": [
            "UI/UX Designer",
            "Graphic Designer",
            "Game Designer"
        ],

        "Healthcare & Social Services": [
            "Doctor",
            "Nurse",
            "Psychologist"
        ],

        "Business & Management": [
            "Business Analyst",
            "Entrepreneur",
            "Marketing Manager"
        ]
    }

    recommended = careers.get(
        interest,
        [
            "Software Developer",
            "Business Analyst",
            "Data Analyst"
        ]
    )

    if aptitude_score >= 8:

        confidence = "High"

    elif aptitude_score >= 5:

        confidence = "Medium"

    else:

        confidence = "Needs Improvement"


    return {
        "careers": recommended,
        "confidence": confidence
    }