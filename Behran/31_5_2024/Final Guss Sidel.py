def gauss_seidel(A, b, initial_guess, max_iterations, tolerance=0.001):
    import numpy as np

    x = np.array(initial_guess, dtype=float)
    n = len(b)
    
    print(f"{'Iteration':<10} {'x':<30} {'Error':<10}")
    
    for k in range(max_iterations):
        x_new = np.copy(x)
        
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        
        # Calculate the error
        error = np.linalg.norm(x_new - x, ord=np.inf)
        
        # Print the current iteration, x values, and error
        print(f"{k+1:<10} {x_new} {error:<10}")
        
        # Check for convergence
        if error < tolerance:
            print(f"Converged after {k+1} iterations")
            return x_new
        
        x = x_new

    print("Maximum iterations reached")
    return x

# Function to get user input and solve the system
def main():
    import numpy as np

    n = int(input("Enter the number of equations: "))

    A = np.zeros((n, n))
    b = np.zeros(n)
    initial_guess = np.zeros(n)
    
    print("Enter the coefficients of the matrix A row by row:")
    for i in range(n):
        A[i] = [float(x) for x in input(f"Row {i+1}: ").split()]
    
    print("Enter the constants vector b:")
    b = [float(x) for x in input().split()]

    print("Enter the initial guess:")
    initial_guess = [float(x) for x in input().split()]
    
    max_iterations = int(input("Enter the maximum number of iterations: "))

    solution = gauss_seidel(A, b, initial_guess, max_iterations)
    print("Solution:", solution)

if __name__ == "__main__":
    main()
