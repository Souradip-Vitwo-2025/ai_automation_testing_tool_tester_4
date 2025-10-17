<?php
/**
 * This script is intentionally flawed for testing the GitHub/SQL Analyzer.
 * It contains:
 * 1. PHP Syntax Error
 * 2. PHP Logical Error (incorrect array indexing/iteration)
 * 3. PHP Runtime Error (Division by zero)
 * 4. A poorly formatted SQL statement to test the SQL Validator.
 */

// --- 1. PHP Syntax Error (Missing semicolon and wrong comparison) ---
function is_prime($number)
{
    if ($number < 2 == true)  // Syntax-like issue (bad comparison)
    {
        return false;
    }

    // Logical inefficiency: basic trial division up to number instead of sqrt(number)
    for ($i = 2; $i < $number; $i++) {
        if ($number % $i == 0) {
            return false;
        }
    }
    return true;
}

// --- 2. PHP Logical Error (Incorrect array access/Iteration) ---
function calculate_average($data_list)
{
    /**
     * Calculates the average of a list of numbers, with an intentional bug.
     */
    $total = 0;

    // BUG: Iterating by index but always accessing a fixed index (1)
    for ($i = 0; $i < count($data_list); $i++) {
        $total += $data_list[1];
    }

    // BUG: Potential DivisionByZeroError if array is empty
    return $total / count($data_list);
}

// --- 3. PHP Runtime Error (DivisionByZero trigger) ---
function process_data($a, $b)
{
    // This function triggers a runtime exception when $b = 0
    echo "Result: " . intdiv($a, $b) . "\n";  // Division by zero error possibility
    return $a / $b;
}

// --- 4. Poorly formatted SQL Query (to test SQL Validator) ---
function get_recent_orders($customer_id)
{
    /**
     * Returns an SQL query string.
     * It contains a table/column typo and a constraint violation (NULL for NOT NULL field).
     */
    // Typos: 'erp_branc_sales_order' (table typo) and 'custumer_id' (column typo)
    // Constraint violation: setting so_number to NULL (it's NOT NULL UNIQUE)
    $sql_query = "
        INSERT INTO erp_branc_sales_orders 
        (so_id, custumers_id, approvalStatus, so_number, createdAt)
        VALUES 
        (1001, {$customer_id}, 'Pending', NULL, '2024-10-15');
    ";
    return $sql_query;
}

// --- Main workflow ---
function main_workflow()
{
    echo "Starting analysis...\n";

    // Triggering Syntax Error (the above function definition has a subtle one)
    $result = is_prime(17);
    echo "Is 17 prime? " . ($result ? "Yes" : "No") . "\n";

    // Triggering the Logical Error
    $data = array(10, 20, 30, 40);
    $avg = calculate_average($data);
    echo "Calculated average (expected 25, actual bug result): {$avg}\n";

    // Triggering the Runtime Error
    // BUG: If the random number is 0, this will crash.
    $divisor = rand(0, 1); // 50% chance of being 0
    try {
        process_data(100, $divisor);
    } catch (DivisionByZeroError $e) {
        echo "Runtime error handled.\n";
    }

    // Execute SQL generation
    $sql_string = get_recent_orders(999);
    echo "\nGenerated Buggy SQL:\n";
    echo $sql_string . "\n";
}

// Execute the main function, which will immediately fail due to the syntax error
main_workflow();
?>
