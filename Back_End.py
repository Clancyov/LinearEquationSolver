from decimal import Decimal, getcontext
import numpy as np

# Set the precision high enough to handle the required number of decimal places
getcontext().prec = 50

class backEnd:
    def jacobi(self, A, b, x0, max_iterations, tol=Decimal('1e-40')):
        n = len(b)
        x = np.array([Decimal(xi) for xi in x0], dtype=object)
        x_new = np.zeros_like(x, dtype=object)
        iteration_details = []

        iteration_details.append({"Iteration": "Initial guess", **{f"x[{i}]": x0[i] for i in range(n)}})
        for k in range(max_iterations):
            details = {}
            for i in range(n):
                s = sum(Decimal(A[i][j]) * x[j] for j in range(n) if i != j)
                x_new[i] = (Decimal(b[i]) - s) / Decimal(A[i][i])
                details[f"x[{i}]"] = x_new[i]
            
            iteration_details.append({"Iteration": k + 1, **details})
            
            if np.linalg.norm([float(x_new[j] - x[j]) for j in range(n)], ord=np.inf) < float(tol):
                iteration_details.append({"Iteration": f"Converged after {k+1} iterations", **{}})
                return x_new, iteration_details
            
            x = x_new.copy()
        
        iteration_details.append({"Iteration": "Maximum iterations reached without convergence", **{}})
        return x, iteration_details

    def gaussSeidel(self, A, b, x0, max_iterations, tol=Decimal('1e-40')):
        n = len(b)
        x = np.array([Decimal(xi) for xi in x0], dtype=object)
        iteration_details = []

        iteration_details.append({"Iteration": "Initial guess", **{f"x[{i}]": x0[i] for i in range(n)}})
        for k in range(max_iterations):
            x_new = np.copy(x)
            details = {}
            
            for i in range(n):
                s1 = sum(Decimal(A[i, j]) * x_new[j] for j in range(i))
                s2 = sum(Decimal(A[i, j]) * x[j] for j in range(i + 1, n))
                x_new[i] = (Decimal(b[i]) - s1 - s2) / Decimal(A[i, i])
                details[f"x[{i}]"] = x_new[i]
            
            iteration_details.append({"Iteration": k + 1, **details})
            
            error = np.linalg.norm([float(x_new[j] - x[j]) for j in range(n)], ord=np.inf)
            if error < float(tol):
                iteration_details.append({"Iteration": f"Converged after {k+1} iterations", **{}})
                return x_new, iteration_details
            
            x = x_new

        iteration_details.append({"Iteration": "Maximum iterations reached without convergence", **{}})
        return x, iteration_details
