from DBManager import get_all_cannabinoids, get_all_terpenes, get_user_sessions, get_all_tags

if __name__ == "__main__":
    # Test fetching cannabinoids and terpenes
    cannabinoids = get_all_cannabinoids()
    terpenes = get_all_terpenes()
    tags = get_all_tags()
    

    # Test fetching user sessions
    user_id = 1  # Replace with a valid user ID from your database
    sessions = get_user_sessions(user_id)
    print(sessions)