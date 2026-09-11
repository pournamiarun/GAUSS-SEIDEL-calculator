from flask import Flask, render_template, request
import math

app = Flask(__name__)


# ---------------------------------------------------------
# GAUSS-SEIDEL CALCULATOR
# ---------------------------------------------------------
def gauss_seidel(A, b, initial_guess, tolerance, max_iterations):
    """
    Solves a 3x3 system of linear equations using
    the Gauss-Seidel iterative method.
    """

    n = len(A)
    x = initial_guess.copy()

    iterations = []

    # Check for zero diagonal elements
    for i in range(n):
        if A[i][i] == 0:
            return {
                "success": False,
                "message": (
                    f"Cannot calculate because the diagonal element "
                    f"A[{i + 1}][{i + 1}] is zero."
                ),
                "iterations": [],
                "solution": None,
                "diagonally_dominant": False
            }

    # Check diagonal dominance
    diagonally_dominant = True

    for i in range(n):
        diagonal = abs(A[i][i])
        other_sum = sum(abs(A[i][j]) for j in range(n) if j != i)

        if diagonal <= other_sum:
            diagonally_dominant = False
            break

    # Perform Gauss-Seidel iterations
    converged = False

    for iteration in range(1, max_iterations + 1):

        old_x = x.copy()

        for i in range(n):
            sum_before = 0
            sum_after = 0

            # Values already updated during this iteration
            for j in range(i):
                sum_before += A[i][j] * x[j]

            # Values not yet updated
            for j in range(i + 1, n):
                sum_after += A[i][j] * old_x[j]

            x[i] = (b[i] - sum_before - sum_after) / A[i][i]

        # Calculate error
        error = max(abs(x[i] - old_x[i]) for i in range(n))

        iterations.append({
            "iteration": iteration,
            "x1": x[0],
            "x2": x[1],
            "x3": x[2],
            "error": error
        })

        # Convergence check
        if error < tolerance:
            converged = True
            break

    if converged:
        message = (
            f"Converged successfully after {len(iterations)} "
            f"iteration(s)."
        )
    else:
        message = (
            f"The method did not converge within the maximum "
            f"of {max_iterations} iterations."
        )

    return {
        "success": converged,
        "message": message,
        "iterations": iterations,
        "solution": x,
        "diagonally_dominant": diagonally_dominant
    }


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    # Default values for the form
    matrix = [
        [10, 2, 1],
        [1, 5, 1],
        [2, 3, 10]
    ]

    rhs = [7, -8, 6]

    initial_guess = [0, 0, 0]

    tolerance = 0.0001
    max_iterations = 100

    if request.method == "POST":

        try:
            # ---------------------------------------------
            # Read matrix values
            # ---------------------------------------------
            matrix = [
                [
                    float(request.form[f"a{i}{j}"])
                    for j in range(3)
                ]
                for i in range(3)
            ]

            # ---------------------------------------------
            # Read RHS vector
            # ---------------------------------------------
            rhs = [
                float(request.form[f"b{i}"])
                for i in range(3)
            ]

            # ---------------------------------------------
            # Read initial guesses
            # ---------------------------------------------
            initial_guess = [
                float(request.form[f"x{i}"])
                for i in range(3)
            ]

            # ---------------------------------------------
            # Read tolerance
            # ---------------------------------------------
            tolerance = float(request.form["tolerance"])

            if tolerance <= 0:
                raise ValueError("Tolerance must be greater than zero.")

            # ---------------------------------------------
            # Read maximum iterations
            # ---------------------------------------------
            max_iterations = int(request.form["max_iterations"])

            if max_iterations <= 0:
                raise ValueError(
                    "Maximum iterations must be greater than zero."
                )

            # ---------------------------------------------
            # Run Gauss-Seidel
            # ---------------------------------------------
            result = gauss_seidel(
                matrix,
                rhs,
                initial_guess,
                tolerance,
                max_iterations
            )

        except ValueError as e:
            error = str(e)

        except Exception as e:
            error = (
                "Something went wrong while processing your input. "
                "Please check all values."
            )

    return render_template(
        "index.html",
        result=result,
        error=error,
        matrix=matrix,
        rhs=rhs,
        initial_guess=initial_guess,
        tolerance=tolerance,
        max_iterations=max_iterations
    )


# ---------------------------------------------------------
# RUN FLASK SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, port=8080)
