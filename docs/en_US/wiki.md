# Lamina Language Wiki

## Advantages
1.  **Precise Mathematical Calculations**: Solves the problem of floating-point precision loss from the ground up. It supports symbolic storage and operations for rational numbers (fractions) and irrational numbers (√, π, e), maintaining precision even after multiple iterative calculations.
2.  **Concise and Intuitive Syntax**: Supports automatic semicolon insertion, omission of parentheses in if/while statements, and shorthand for parameter-less functions, reducing code redundancy and aligning with mathematical expression habits.
3.  **Native Math-Friendliness**: Natively supports mathematical operations such as vectors, matrix operations, and large integer factorials without requiring third-party libraries, meeting the needs of complex mathematical problems.
4.  **Friendly Development Experience**: The interactive REPL supports keyword highlighting and auto-completion. It provides a full stack trace for errors, facilitating debugging. The smart terminal automatically adapts colors to avoid garbled text.
5.  **Modular Design**: Incorporates external modules using the `include` statement and supports the `::` namespace accessor, enabling code reuse and isolation.
6.  **Flexible Data Types**: Covers precise numerical types (rational/irrational), composite types (arrays/matrices/structs), and anonymous functions, adapting to various development scenarios.


## Basic Syntax
### 1. Variable Declaration
**Template**:
```lamina
// Normal variable declaration
var var_name = expression;
// Big integer variable declaration (arbitrary precision)
bigint big_var_name = expression;
```
**Example**:
```lamina
var pi = 3.1415; // Declare a floating-point variable pi
var score = 95; // Declare an integer variable score
var half = 1/2; // Declare a rational number variable half (automatically stored as a fraction)
bigint fact_25 = 25!; // Declare a big integer variable to store the factorial of 25
```

### 2. Comments
**Template**:
```lamina
// Single-line comment: comment content
/* 
   Block comment:
   Supports multi-line text
   Suitable for long explanatory comments
*/
```
**Example**:
```lamina
// This is a single-line comment to explain the function of the next line of code
var radius = 5; // Declare a variable for the circle's radius

/* 
   Block comment example:
   The following code calculates the area of a circle
   It depends on the pi() function to get the precise value of pi
*/
var area = pi() * radius ^ 2;
```

### 3. Conditional Statements (if-else)
**Template**:
```lamina
if condition {
    // Code to be executed if the condition is true
} else {
    // Code to be executed if the condition is false
}
```
**Example**:
```lamina
var num = 8;
if num > 0 {
    print("num is a positive number"); // This will be printed as the condition is true
} else {
    print("num is a non-positive number");
}
```

### 4. Loop Statements
#### (1) while loop
**Template**:
```lamina
while condition {
    // Code to be executed repeatedly as long as the condition is true
}
```
**Example**:
```lamina
var count = 1;
while count <= 3 {
    print("Current count: ", count); // Prints 1, 2, 3 in sequence
    count = count + 1;
}
```

#### (2) loop (infinite loop)
#### (2) loop (counted loop)

Note: The old infinite `loop { }` form has been removed. `loop` must now be followed by a count expression.

**Template**:

```lamina
loop <count-expression> {
    // Code executed count-expression times
}
```

Description: `<count-expression>` may be an integer literal, a variable, or any expression that evaluates to an integer at runtime. Accepted numeric types at runtime include: int, bigint (within representable range), rational (only when denominator == 1), and floats only when they are integer-valued (e.g., `3.0`). Irrational numbers, symbolic expressions, strings, arrays, etc. are not valid counts.

Error behavior:
- If the count-expression evaluates to a non-integer (e.g., 2.5 or 1/2), a runtime error "Repeat count must be an integer" is thrown.
- If the count is negative, a runtime error "Repeat count must be non-negative" is thrown.
- If the count is too large to be represented as an integer for looping, a runtime error like "Repeat count too large" or "Repeat count out of range" is thrown.

**Examples**:

```lamina
var i = 0;
loop 3 { i = i + 1; };
print(i); // prints 3

var n = 4;
var acc = 0;
loop n { acc = acc + 1; };
print(acc); // prints 4

var a = 2; var b = 3;
loop (a + b) { print("hi"); } // equivalent to loop 5 { ... }
```

### 5. Function Definition
#### (1) Parameter-less Function
**Template**:
```lamina
func func_name {
    // Function body code
    return return_value; // Optional, can be omitted if there is no return value
}
```
**Example**:
```lamina
func say_hello {
    print("Hello, Lamina!"); // No return value, just prints output
}
say_hello(); // Call the function, outputs "Hello, Lamina!"
```

#### (2) Function with Parameters
**Template**:
```lamina
func func_name(param1, param2) {
    // Logic code based on parameters
    return result; // Return the calculated result
}
```
**Example**:
```lamina
func add(a, b) {
    return a + b; // Returns the sum of a and b
}
var sum = add(3, 5); // Call the function, sum = 8
print("3 + 5 =", sum);
```

### 6. Anonymous Functions
**Template**:
```lamina
var func_var = do |param1, param2| {
    // Function body code
    return result;
};

// Can be shortened for single-expression functions
var simple_func = |param1, param2| expression;
```
**Example**:
```lamina
var multiply = do |x, y| {
    return x * y; // Anonymous function to perform multiplication
};
var product = multiply(4, 6); // Call the anonymous function, product = 24
print("4 * 6 =", product);

var add = |a,b| a+b;
var result = add(1,33)
print("1 + 33 =", result)
```

### 7. Struct Declaration
**Template**:
```lamina
var struct_name = {
    key1 = value1;
    key2 = value2;
    // More key-value pairs...
};```
**Example**:
```lamina
var student = {
    name = "Tom";
    age = 15;
    scores = [90, 85, 92]; // Struct with a nested array
};
```

### 8. Struct Member Access
**Template**:
```lamina
struct_name.key; // Access a specific member of the struct
```
**Example**:
```lamina
var student = { name = "Tom"; age = 15; };
print("Student Name: ", student.name); // Outputs "Student Name: Tom"
print("Student Age: ", student.age); // Outputs "Student Age: 15"
student.age = 16
```

### 9. Module Import
**Template**:
```lamina
include "module_path"; // Imports the module from the specified path, .lm extension is automatically appended
```
**Example**:
```lamina
include "math_utils"; // Import the math utilities module
include "lib/vectors"; // Import the vector calculation module (relative path)
var root = math::sqrt(2); // Access a module function using the namespace
```

### 10. Line Continuation Character
**Template**:
```lamina
var long_expression = expression1 + expression2 + \
                      expression3 + expression4; // \ is used to split long expressions
```
**Example**:
```lamina
var total = 10 + 20 + 30 + \
            40 + 50; // Equivalent to var total = 10+20+30+40+50;
print("Sum: ", total); // Outputs "Sum: 150"
```


## Basic Types
1.  **int**: Standard integer type, supporting arithmetic operations on positive and negative integers. The type does not need to be explicitly declared.
    Example:
    ```lamina
    var a = 42; // Positive integer
    var b = -10; // Negative integer
    var c = a + b; // c = 32 (int type)
    ```

2.  **float**: Floating-point number type, used for compatibility with traditional floating-point operations. It has precision limitations.
    Example:
    ```lamina
    var pi_approx = 3.14; // Floating-point number
    var temp = -0.5; // Negative floating-point number
    ```

3.  **rational**: Precise rational number type. It automatically stores division results as fractions to avoid precision loss and supports automatic simplification.
    Example:
    ```lamina
    var frac1 = 16/9; // Stored as 16/9, not 1.777...
    var frac2 = 4/6; // Automatically simplified to 2/3
    var sum_frac = frac1 + frac2; // Precise calculation, result is 34/9
    ```

4.  **irrational**: Precise irrational number type. Stores numbers in symbolic form (e.g., √, π, e) and supports symbolic operations and simplification.
    Example:
    ```lamina
    var root2 = sqrt(2); // Stored as √2
    var root8 = sqrt(8); // Automatically simplified to 2√2
    var pi_val = pi(); // Stored as π
    var product = root2 * root2; // Result is 2 (int type)
    ```

5.  **bool**: Boolean type, which includes only two values: `true` and `false`, used for conditional checks.
    Example:
    ```lamina
    var is_pass = true;
    var is_empty = false;
    if is_pass {
        print("Test passed");
    }
    ```

6.  **string**: String type, with text content enclosed in double quotes. Supports string-related function operations.
    Example:
    ```lamina
    var greeting = "Hello, Lamina!";
    var name = "Alice";
    ```

7.  **null**: Null value type, representing only the value `null`. Used to indicate that a variable is unassigned or a function has no return value.
    Example:
    ```lamina
    var empty_var = null;
    func no_return() {
        print("No return value");
        return null; // Explicitly return null
    }
    ```

8.  **bigint**: Arbitrary-precision large integer type. Must be explicitly declared. Supports extra-large integers (e.g., for factorials, large-value calculations).
    Example:
    ```lamina
    bigint large_num = 999999999999999; // Extra-large integer
    bigint fact_30 = 30!; // Factorial of 30 (a large integer result)
    ```

9.  **array**: Array type, with elements enclosed in square brackets. Supports index access and array-related functions.
    Example:
    ```lamina
    var scores = [90, 85, 92]; // One-dimensional array
    var names = ["Tom", "Alice", "Bob"]; // String array
    ```

10. **matrix**: Matrix type, represented by nested arrays (two-dimensional arrays). Supports operations like matrix determinant, multiplication, etc.
    Example:
    ```lamina
    var mat2x2 = [[1, 2], [3, 4]]; // 2x2 matrix
    var mat3x1 = [[1], [2], [3]]; // 3x1 column matrix
    ```

11. **struct**: Struct type, with key-value pairs enclosed in curly braces. Supports custom members and member access.
    Example:
    ```lamina
    var person = {
        name = "Bob";
        age = 20;
        is_student = true;
    };
    ```

12. **lambda**: Anonymous function type, used to represent unnamed functions. Can be assigned to variables or passed as arguments.
    Example:
    ```lamina
    var subtract = |a, b| a - b ; // lambda type variable
    ```

13. **module**: Module type, imported using `include`. Contains functions and variables defined within the module, accessible via the `::` namespace.
    Example:
    ```lamina
    include random; // Import the module, module type
    var result = random::random(); // Access a module function
    ```


## Libraries
### 1. Math Functions
-   **Square Root Function**: Calculates the precise square root of a number. Returns an int if it's a perfect square, otherwise returns an irrational.
    ```lamina
    sqrt(x) -> int/irrational
    ```
-   **Pi Function**: Returns the precise symbol for pi, π, of type irrational.
    ```lamina
    pi() -> irrational
    ```-   **Natural Constant Function**: Returns the precise symbol for the natural constant e, of type irrational.
    ```lamina
    e() -> irrational
    ```
-   **Sine Function**: Calculates the sine of an angle. Supports precise numerical input and returns a result with corresponding precision.
    ```lamina
    sin(x) -> rational/irrational/float
    ```
-   **Cosine Function**: Calculates the cosine of an angle. Supports precise numerical input and returns a result with corresponding precision.
    ```lamina
    cos(x) -> rational/irrational/float
    ```
-   **Absolute Value Function**: Returns the absolute value of the input number, maintaining the original type.
    ```lamina
    abs(x) -> int/float/rational/irrational
    ```
-   **Natural Logarithm Function**: Calculates the natural logarithm (base e) of a number, returning a result with corresponding precision.
    ```lamina
    log(x) -> rational/irrational/float
    ```
-   **Factorial Function**: Calculates the factorial of a non-negative integer. Supports int and bigint types as input and returns the corresponding integer type.
    ```lamina
    factorial(n) -> int/bigint
    ```

### 2. Vector/Matrix Functions
-   **Vector Dot Product Function**: Calculates the dot product of two vectors of the same dimension, returning a numerical type (int/rational, etc.).
    ```lamina
    dot(v1: array, v2: array) -> int/rational/float
    ```
-   **Vector Cross Product Function**: Calculates the cross product of two three-dimensional vectors, returning a new three-dimensional array (matrix).
    ```lamina
    cross(v1: array, v2: array) -> array
    ```
-   **Vector Norm Function**: Calculates the norm (length) of a vector, returning a precise numerical type (irrational/rational, etc.).
    ```lamina
    norm(v: array) -> rational/irrational/float
    ```
-   **Matrix Determinant Function**: Calculates the determinant of a two-dimensional square matrix, returning a numerical type (int/rational, etc.).
    ```lamina
    det(mat: array) -> int/rational/float
    ```

### 3. Utility Functions
-   **Print Function**: Outputs one or more items to the console, automatically adding a newline at the end. Has no return value.
    ```lamina
    print(...) -> null
    ```
-   **Input Function**: Displays a prompt text in the console and gets user input, returning it as a string type.
    ```lamina
    input(prompt: string) -> string/float
    ```
-   **Decimal to Fraction Function**: Converts a floating-point number to a precise rational number (fraction), with automatic simplification.
    ```lamina
    fraction(x: float) -> rational
    ```
-   **Fraction to Decimal Function**: Converts a rational number (fraction) to a floating-point number, with support for desired precision.
    ```lamina
    decimal(x: rational) -> float
    ```
-   **Get Type Function**: Returns the type name of a variable as a string.
    ```lamina
    typeof(x) -> string
    ```
-   **Deep Copy Function**: Performs a deep copy of composite types like structs, arrays, anonymous functions, and modules. Modifying the copy does not affect the original object.
    ```lamina
    copy(x) -> any
    ```
-   **Get Size Function**: Returns the length of an array or the number of members in a struct, as an int type.
    ```lamina
    size(x: array/struct) -> int
    ```
-   **Assert Function**: Checks if a condition is true. If it's false, it throws an error and displays a custom message.
    ```lamina
    assert(condition: bool, msg: string = "") -> null
    ```

### 4. Array Functions
-   **Array ForEach Function**: Iterates over each element of an array and executes a specified function on it. Has no return value.
    ```lamina
    foreach(arr: array, func: lambda) -> null
    ```
-   **Array Map Function**: Iterates over each element of an array, processes it with a specified function, and returns a new array.
    ```lamina
    map(arr: array, func: lambda) -> array
    ```
-   **Array Find Function**: Finds the first element in an array that satisfies a condition and returns its value or null.
    ```lamina
    find(arr: array, func: lambda) -> any/null
    ```
-   **Array Replace Function**: Replaces elements in an array that satisfy a condition. Returns null.
    ```lamina
    replace(arr: array, func: lambda, new_val) -> null
    ```

### 5. String Functions
-   **String Concatenation Function**: Concatenates multiple strings and returns the new concatenated string.
    ```lamina
    string::cat(...) -> string
    ```
-   **Get Character Function**: Gets the character at a specified index.
    ```lamina
    string::at(index: int) -> int
    ```
-   **Substring Function**: Extracts a substring of a specified length starting from a given index, returning a new string.
    ```lamina
    string::sub(str: string, start_index: int, len: int) -> string
    ```
### 6. Random Functions
-   **Random Float Function**: Returns a random floating-point number between 0 (inclusive) and 1 (exclusive).
    ```lamina
    random::random() -> float
    ```
-   **Random Integer Function**: Returns a random integer within the range `[start, end]` (inclusive of both boundaries).
    ```lamina
    random::randint(start: int, end: int) -> int
    ```
-   **Random String Function**: Randomly selects a character from the input string and returns that character (string type).
    ```lamina
    random::randstr(chars: string) -> string
    ```

### 7. Time Functions
-   **Get Time Function**: Returns the current system time in "HH:MM:SS" format, as a string type.
    ```lamina
    time::time() -> string
    ```
-   **Get Date Function**: Returns the current system date in "YYYY-MM-DD" format, as a string type.
    ```lamina
    time::date() -> string
    ```
-   **Format Date Function**: Converts a date string to a specified format, returning the formatted date string.
    ```lamina
    time::format_date(date: string, format: string = "YYYY-MM-DD") -> string
    ```

### 8. CAS-related Functions
-   **CAS Parse Function**: Parses a mathematical expression and converts it into an internal format that CAS can handle, returning the parsed CAS object.
    ```lamina
    cas::parse(expr: string) -> cas_object
    ```
-   **CAS Simplify Function**: Simplifies the mathematical expression represented by a CAS object, returning the simplified CAS object.
    ```lamina
    cas::simplify(cas_obj: cas_object) -> cas_object
    ```
-   **CAS Differentiate Function**: Differentiates the function represented by a CAS object, returning the resulting CAS object.
    ```lamina
    cas::differentiate(cas_obj: cas_object, var: string) -> cas_object
    ```
-   **CAS Evaluate Function**: Performs numerical or symbolic calculation on the expression represented by a CAS object, returning the result (either a CAS object or a basic data type).
    ```lamina
    cas::evaluate(cas_obj: cas_object) -> any
    ```
-   **CAS Store Function**: Associates a specified name with a CAS object and stores it for later use. Has no return value or returns a boolean indicating success.
    ```lamina
    cas::store(name: string, cas_obj: cas_object) -> bool/void
    ```-   **CAS Load Function**: Loads a previously stored CAS object by its name, returning the corresponding CAS object (or null if it doesn't exist).
    ```lamina
    cas::load(name: string) -> cas_object/null
    ```
-   **CAS Evaluate At Function**: Evaluates the expression represented by a CAS object at a specific variable value, returning the numerical result.
    ```lamina
    cas::evaluate_at(cas_obj: cas_object, var: string, value: number) -> number
    ```
-   **CAS Linear Solve Function**: Solves a linear equation or a system of linear equations (passed in as a CAS object), returning the solution (as a CAS object or a set of solutions).
    ```lamina
    cas::solve_linear(cas_obj: cas_object) -> cas_object/array
    ```
-   **CAS Numerical Derivative Function**: Numerically differentiates a function (represented by a CAS object), returning the numerical derivative result.
    ```lamina
    cas::numerical_derivative(cas_obj: cas_object, var: string, value: number) -> number
    ```


### 9. IO Functions
1.  **Read File**: Reads the content of a file based on the specified path and matching pattern, returning the read string data.
```lamina
fast_io::read(path, pattern) -> string
```
2.  **Write to File**: Writes content to a file at the specified path (Note: Assuming the function includes a content parameter as per standard logic), returning a boolean indicating success or failure.
```lamina
fast_io::write(path, content) -> bool
```
3.  **Create File**: Creates a new file at the specified path and writes initial content to it, returning a boolean indicating success or failure.
```lamina
fast_io::create(path, init_content) -> bool
```

### 10. Program Control Functions
-   **Exit Program Function**: Terminates the current program's execution, with an optional exit code (0 for normal exit, non-zero for abnormal).
    ```lamina
    exit(code: int = 0) -> null
    ```
-   **To String Function**: Converts a variable of any type to a string type, returning the converted string.
    ```lamina
    to_string(x) -> string
    ```
-   **System Execute Function**: Executes a system command.
```
system(cmd: string) -> null
```

### 11. Variable and Function Query Functions
-   **All Scopes Variable Query Function**: Returns a list of the names of all current global variables, as an array type.
    ```lamina
    vars() -> array
    ```
-   **Local Variables Query Function**: Returns a list of the names of all local variables in the current scope, as a struct type.
    ```lamina
    locals() -> struct
    ```-   **Global Variables Query Function**: Returns a list of the names of variables in the global scope, as a struct type.
    ```lamina
    globals() -> struct
    ```