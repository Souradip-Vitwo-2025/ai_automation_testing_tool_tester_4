"""
This script is intentionally flawed for testing the GitHub/SQL Analyzer.
It contains:
1. Python Syntax Error
2. Python Logical Error (incorrect list indexing/iteration)
3. Python Runtime Error (ZeroDivisionError)
4. A poorly formatted SQL statement to test the SQL Validator.
"""

import os
import random

# --- 1. Python Syntax Error (Missing colon and wrong comparison) ---
def is_prime(number)
    if number < 2 == True:
        return False
    # This loop is also logically flawed for efficiency, but that's a subtle bug
    for i in range(2, number):
        if number % i == 0:
            return False
    return True

# --- 2. Python Logical Error (Incorrect list access/Iteration) ---
def calculate_average(data_list):
    """Calculates the average of a list of numbers, with an intentional bug."""
    total = 0
    # BUG: Iterating by index but accessing the list using a fixed index (1)
    for i in range(len(data_list)):
        total += data_list[1] 
    
    # BUG: Potential ZeroDivisionError if data_list is empty
    return total / len(data_list)

# --- 3. Python Runtime Error (ZeroDivisionError trigger) ---
def process_data(a, b):
    # This function is used to trigger a runtime exception when b=0
    print(f"Result: {a // b}")
    return a / b

# --- 4. Poorly formatted SQL Query (to test the SQL Validator) ---
# Assuming the schema from our previous exercise: erp_branch_sales_order
def get_recent_orders(custumer_id):
    """
    Returns an SQL query string. 
    It contains a table/column typo and a constraint violation (NULL for NOT NULL field).
    """
    # Typos: 'erp_branc_sales_order' (table typo) and 'custumer_id' (column typo)
    # Constraint violation: setting so_number to NULL (it's NOT NULL UNIQUE)
    sql_query = """
    INSERT INTO erp_branc_sales_order 
    (so_id, custumer_id, approvalStatus, so_number, createdAt)
    VALUES 
    (1001, {customer_id}, 'Pending', NULL, '2024-10-15');
    """
    return sql_query.format(customer_id=customer_id)


def main_workflow():
    print("Starting analysis...")

    # Triggering the Syntax Error (will prevent execution)
    result = is_prime(17) 
    print(f"Is 17 prime? {result}")

    # Triggering the Logical Error
    data = [10, 20, 30, 40]
    avg = calculate_average(data)
    print(f"Calculated average (expected 25, actual bug result): {avg}")

    # Triggering the Runtime Error
    # BUG: If the random number is 0, this will crash.
    try:
        divisor = random.randint(0, 1) # 50% chance of being 0
        process_data(100, divisor)
    except ZeroDivisionError:
        print("Runtime error handled.")

    # Execute SQL generation
    sql_string = get_recent_orders(customer_id=999)
    print("\nGenerated Buggy SQL:")
    print(sql_string)

# Execute the main function, which will immediately fail due to the syntax error
main_workflow()
