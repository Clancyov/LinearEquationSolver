import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import numpy as np
from decimal import Decimal
from Back_End import backEnd

class streamlitGUI:
    methodsList = ["Jacobi", "Gauss-Seidel"]
    backend = backEnd()

    def editable_matrix_input(self, n):
        data = np.zeros((n, n))
        df = pd.DataFrame(data, columns=[f'Col {i}' for i in range(n)], index=[f'Row {i}' for i in range(n)])

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True)
        gb.configure_grid_options(enableRangeSelection=True, headerHeight=30)
        grid_options = gb.build()
        st.write("Matrix Input:")
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=False,
            height=150
        )

        updated_df = grid_response['data']
        return updated_df.values

    def validation(self, toCheck, n):
        try:
            numbers = list(map(Decimal, toCheck.split()))
            if len(numbers) != n:
                return False
            return True
        except ValueError:
            return False

    def GUI(self):
        st.title("Linear Equation Solver")

        method = st.selectbox("Choose Method:", self.methodsList)

        if method == self.methodsList[0]:
            st.title("Jacobi")
        elif method == self.methodsList[1]:
            st.title("Gauss-Seidel")

        col1, col2 = st.columns([2, 2])

        if method == self.methodsList[0]:
            with col1:
                numberOfEquations = st.number_input("Number of Equations:", min_value=2, max_value=10)
                constantVectorB = st.text_input(f"Enter '{numberOfEquations}' elements of the constant vector b (separated by space):")
                if constantVectorB and not self.validation(constantVectorB, numberOfEquations):
                    st.error(f"Please enter the values like: x x or x x x or ...")

            with col2:
                matrix = self.editable_matrix_input(numberOfEquations)
            initialGuess = st.text_input(f"Enter '{numberOfEquations}' elements of initial guess (separated by space):")
            if initialGuess and not self.validation(initialGuess, numberOfEquations):
                st.error(f"Please enter the values like: x x or x x x or ...")
            numberOfIterations = st.number_input("Number of iterations:", min_value=1, max_value=100)
            
            if st.button("Solve"):
                if self.validation(constantVectorB, numberOfEquations) and self.validation(initialGuess, numberOfEquations):
                    A = np.array(matrix, dtype=object)
                    b = np.array(list(map(Decimal, constantVectorB.split())), dtype=object)
                    x0 = np.array(list(map(Decimal, initialGuess.split())), dtype=object)
                    solution, iteration_details = self.backend.jacobi(A, b, x0, numberOfIterations)
                    
                    st.write("Solution:", solution)
                    
                    # Extract and display the convergence message
                    convergence_message = iteration_details.pop()
                    st.title(convergence_message["Iteration"])
                    
                    # Prepare data for display, dynamically getting keys but excluding 'Iteration'
                    columns = sorted(key for key in iteration_details[0].keys() if key != 'Iteration')
                    iteration_data = []
                    for detail in iteration_details:
                        iteration_data.append({col: detail[col] for col in columns})
                    
                    iteration_df = pd.DataFrame(iteration_data)
                    st.table(iteration_df)

        elif method == self.methodsList[1]:
            with col1:
                numberOfRowsColumns = st.number_input("Number of Rows/Columns:", min_value=2, max_value=10)
                VectorB = st.text_input(f"Enter '{numberOfRowsColumns}' elements of the vector b (separated by space):")
                if VectorB and not self.validation(VectorB, numberOfRowsColumns):
                    st.error(f"Please enter the values like: x x or x x x or ...")

            with col2:
                matrix = self.editable_matrix_input(numberOfRowsColumns)
            vectorX0 = st.text_input(f"Enter '{numberOfRowsColumns}' elements of vector X0 (separated by space):")
            if vectorX0 and not self.validation(vectorX0, numberOfRowsColumns):
                st.error(f"Please enter the values like: x x or x x x or ...")
            numberOfIterations = st.number_input("Number of iterations:", min_value=1, max_value=100)
            
            if st.button("Solve"):
                if self.validation(VectorB, numberOfRowsColumns) and self.validation(vectorX0, numberOfRowsColumns):
                    A = np.array(matrix, dtype=object)
                    b = np.array(list(map(Decimal, VectorB.split())), dtype=object)
                    x0 = np.array(list(map(Decimal, vectorX0.split())), dtype=object)
                    solution, iteration_details = self.backend.gaussSeidel(A, b, x0, numberOfIterations)
                    
                    st.write("Solution:", solution)
                    
                    # Extract and display the convergence message
                    convergence_message = iteration_details.pop()
                    st.title(convergence_message["Iteration"])
                    
                    # Prepare data for display, dynamically getting keys but excluding 'Iteration'
                    columns = sorted(key for key in iteration_details[0].keys() if key != 'Iteration')
                    iteration_data = []
                    for detail in iteration_details:
                        iteration_data.append({col: detail[col] for col in columns})
                    
                    iteration_df = pd.DataFrame(iteration_data)
                    st.table(iteration_df)

ss = streamlitGUI()
ss.GUI()
