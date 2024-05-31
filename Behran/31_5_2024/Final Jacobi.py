import numpy as np

def jacobi(A, b, x0, max_iterations, tol=1e-10):
    n = len(b)
    x = x0.copy()
    x_new = np.zeros_like(x)
    
    print(f"Initial guess: {x0}")
    for k in range(max_iterations):
        print(f"\nIteration {k+1}")
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i][i]
            print(f"x[{i}] = ({b[i]} - {s}) / {A[i][i]} = {x_new[i]}")
        
        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"\nConverged after {k+1} iterations")
            return x_new
        
        x = x_new.copy()
    
    print("\nMaximum iterations reached without convergence")
    return x

def input_matrix():
    print("Enter the elements of matrix A row-wise, separated by space:")
    rows = int(input("Enter the number of rows/columns (n): "))
    A = []
    for i in range(rows):
        A.append(list(map(float, input(f"Row {i+1}: ").split())))
    return np.array(A, dtype=float)

def input_vector(n):
    print("Enter the elements of vector b, separated by space:")
    b = list(map(float, input().split()))
    return np.array(b, dtype=float)

def input_initial_guess(n):
    print("Enter the initial guess vector x0, separated by space:")
    x0 = list(map(float, input().split()))
    return np.array(x0, dtype=float)

def main():
    A = input_matrix()
    n = A.shape[0]
    b = input_vector(n)
    x0 = input_initial_guess(n)
    max_iterations = int(input("Enter the maximum number of iterations: "))
    
    solution = jacobi(A, b, x0, max_iterations)
    print("Solution:", solution)

if __name__ == "__main__":
    main()
