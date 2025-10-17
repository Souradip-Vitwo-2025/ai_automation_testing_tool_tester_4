import sqlite3

# -------------------------
# Function with table/column typos + datatype mismatch
# -------------------------
def get_user_orders(user_id):
    conn = sqlite3.connect("db.sqlite3")
    cursor = conn.cursor()

    # Intentional typo: "usernamee" instead of "username" and "user" instead of "users"
    cursor.execute("SELECT id, usernamee FROM user WHERE id = ?", (user_id,))
    results = cursor.fetchall()

    conn.close()
    return results


# -------------------------
# Function with correct SQL but will trigger UNIQUE constraint violation
# -------------------------
def insert_user(username, email):
    conn = sqlite3.connect("db.sqlite3")
    cursor = conn.cursor()

    # Insert a duplicate email to trigger UNIQUE constraint violation
    cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", (username, email))
    conn.commit()

    conn.close()


# -------------------------
# Function with datatype mismatch + NOT NULL violation
# -------------------------
def insert_order(user_id, product, amount):
    conn = sqlite3.connect("db.sqlite3")
    cursor = conn.cursor()

    # Intentional mistake: amount as text, product as NULL
    cursor.execute(
        "INSERT INTO orders (user_id, product, amount) VALUES (?, ?, ?)",
        (user_id, None, "eight hundred")
    )
    conn.commit()

    conn.close()


# -------------------------
# Function with FOREIGN KEY violation
# -------------------------
def insert_order_with_fk_violation():
    conn = sqlite3.connect("db.sqlite3")
    cursor = conn.cursor()

    # user_id = 999 does not exist → FK violation
    cursor.execute(
        "INSERT INTO orders (user_id, product, amount) VALUES (?, ?, ?)",
        (999, "Tablet", 300.00)
    )
    conn.commit()

    conn.close()


# -------------------------
# Function with UPDATE causing NOT NULL + datatype issues
# -------------------------
def update_user_email(user_id, new_email):
    conn = sqlite3.connect("db.sqlite3")
    cursor = conn.cursor()

    # Intentional NULL email → violates NOT NULL, datatype mismatch
    cursor.execute(
        "UPDATE users SET email = ? WHERE id = ?",
        (None, user_id)
    )
    conn.commit()

    conn.close()


# -------------------------
# MAIN — Run test cases
# -------------------------
if __name__ == "__main__":
    print("Test functions ready. Run your SQL validator on these functions.")
