import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import numpy as np

class streamlitGUI:
    methodsList = ["Jacobi", "Gauss-Seidel"]

    def editable_matrix_input(self, n):
        data = np.zeros((n, n))
        df = pd.DataFrame(data, columns=[f'Col {i}' for i in range(n)], index=[f'Row {i}' for i in range(n)])

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True)
        gb.configure_grid_options(enableRangeSelection=True, headerHeight=30)  # Adjust header height as needed
        grid_options = gb.build()
        st.write("Matrix Input:")
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=False,
            height=150  # Use 'auto' for dynamic height
        )

        updated_df = grid_response['data']
        return updated_df.values

    def GUI(self):
        st.title("Linear Equation Solver")

        method = st.selectbox("Choose Method:", self.methodsList)

        # Show title based on selected method
        if method == self.methodsList[0]:
            st.title("Jacobi")
        elif method == self.methodsList[1]:
            st.title("Gauss-Seidel")

        # Create a two-column layout
        col1, col2 = st.columns([2, 2])

        if method == self.methodsList[0]:
            # Left column for number of equations
            with col1:
                numberOfEquations = st.number_input("Number of Equations:", min_value=2, max_value=10)
                constantVectorB = st.text_input(f"Enter '{numberOfEquations}' elements of the constant vector b (separated by space):")
            # Right column for matrix input
            with col2:
                matrix = self.editable_matrix_input(numberOfEquations)
            initialGuess = st.text_input(f"Enter '{numberOfEquations}' elements of initial guess (separated by space):")
            numberOfIterations = st.number_input("Number of iterations:", min_value=1, max_value=100)
        elif method == self.methodsList[1]:
            pass  # Currently no GUI elements for Gauss-Seidel

ss = streamlitGUI()
ss.GUI()
