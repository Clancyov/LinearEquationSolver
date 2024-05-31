import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

def jacobi(A, b, x0, max_iterations, tol=1e-10):
    n = len(b)
    x = x0.copy()
    x_new = np.zeros_like(x)
    steps = []
    
    for k in range(max_iterations):
        step_details = [k + 1]
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i][i]
            step_details.append(x_new[i])
        
        error = np.linalg.norm(x_new - x, ord=np.inf)
        step_details.append(error)
        steps.append(step_details)
        
        if error < tol:
            steps.append([k + 1] + list(x_new) + [error])
            return x_new, steps
        
        x = x_new.copy()
    
    steps.append([max_iterations] + list(x_new) + [np.nan])
    return x_new, steps

def gauss_seidel(A, b, initial_guess, max_iterations, tolerance=0.001):
    x = np.array(initial_guess, dtype=float)
    n = len(b)
    steps = []
    
    for k in range(max_iterations):
        x_new = np.copy(x)
        step_details = [k + 1]
        
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
            step_details.append(x_new[i])
        
        error = np.linalg.norm(x_new - x, ord=np.inf)
        step_details.append(error)
        steps.append(step_details)
        
        if error < tolerance:
            steps.append([k + 1] + list(x_new) + [error])
            return x_new, steps
        
        x = x_new
    
    steps.append([max_iterations] + list(x_new) + [np.nan])
    return x_new, steps

def export_to_excel(df, method):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write the method at the top
        method_df = pd.DataFrame([[method]], columns=['Method'])
        method_df.to_excel(writer, index=False, sheet_name='Results', startrow=0)
        df.to_excel(writer, index=False, sheet_name='Results', startrow=2)
    output.seek(0)
    return output

def export_to_pdf(df, method):
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    elements.append(Paragraph(method, styles['Title']))
    
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)
    
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
    
    table.setStyle(style)
    elements.append(table)
    doc.build(elements)
    output.seek(0)
    return output

# Streamlit app
st.title("Numerical Methods Solver")

if 'page' not in st.session_state:
    st.session_state.page = 'input'

if 'method' not in st.session_state:
    st.session_state.method = "Jacobi"
if 'A_input' not in st.session_state:
    st.session_state.A_input = "4 1 2\n1 3 2\n1 2 4"
if 'b_input' not in st.session_state:
    st.session_state.b_input = "4 5 6"
if 'x0_input' not in st.session_state:
    st.session_state.x0_input = "0 0 0"
if 'max_iterations' not in st.session_state:
    st.session_state.max_iterations = 100
if 'show_details' not in st.session_state:
    st.session_state.show_details = False

if st.session_state.page == 'input':
    method = st.selectbox("Choose Method:", ["Jacobi", "Gauss-Seidel"], index=0 if st.session_state.method == "Jacobi" else 1)
    A_input = st.text_area("Matrix A", st.session_state.A_input)
    b_input = st.text_input("Vector b (space-separated):", st.session_state.b_input)
    x0_input = st.text_input("Initial Guess x0 (space-separated):", st.session_state.x0_input)
    max_iterations = st.number_input("Maximum Iterations:", min_value=1, value=st.session_state.max_iterations)

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Calculate"):
            try:
                A = np.array([list(map(float, row.split())) for row in A_input.split('\n')])
                b = np.array(list(map(float, b_input.split())))
                x0 = np.array(list(map(float, x0_input.split())))

                if method == "Jacobi":
                    solution, steps = jacobi(A, b, x0, max_iterations)
                else:
                    solution, steps = gauss_seidel(A, b, x0, max_iterations)

                columns = ['Iteration'] + [f'x[{i}]' for i in range(len(b))] + ['Error']
                df = pd.DataFrame(steps, columns=columns)
                
                st.session_state.method = method
                st.session_state.A_input = A_input
                st.session_state.b_input = b_input
                st.session_state.x0_input = x0_input
                st.session_state.max_iterations = max_iterations
                st.session_state.results = df
                st.session_state.solution = np.round(solution, 5)
                st.session_state.page = 'results'
                st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        if st.button("Details"):
            st.session_state.show_details = not st.session_state.show_details

    if st.session_state.show_details:
        st.write("This program is written by Behran Saedi , Email: behransaedi@gmail.com")

elif st.session_state.page == 'results':
    df = st.session_state.results
    solution = st.session_state.solution

    st.table(df.drop(columns=['Iteration']).round(5))
    st.markdown(f"## Solution: {solution}")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Export to Excel"):
            excel_data = export_to_excel(df, st.session_state.method)
            st.download_button(
                label="Download Excel file",
                data=excel_data,
                file_name='results.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
    
    with col2:
        if st.button("Export to PDF"):
            pdf_data = export_to_pdf(df, st.session_state.method)
            st.download_button(
                label="Download PDF file",
                data=pdf_data,
                file_name='results.pdf',
                mime='application/pdf'
            )
    
    with col3:
        if st.button("Return"):
            st.session_state.page = 'input'
            st.experimental_rerun()

    with col4:
        if st.button("Details"):
            st.session_state.show_details = not st.session_state.show_details

    if st.session_state.show_details:
        st.write("This program is written by Behran Saedi , Email: behransaedi@gmail.com")
