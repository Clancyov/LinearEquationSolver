import sys
import numpy as np
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QLineEdit, QComboBox, QSpinBox, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt


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


class NumericalMethodsApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Numerical Methods Solver')
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.method_label = QLabel('Choose Method:')
        self.method_combo = QComboBox()
        self.method_combo.addItems(['Jacobi', 'Gauss-Seidel'])
        self.method_combo.setStyleSheet("background-color: purple; color: white;")

        self.A_label = QLabel('Matrix A')
        self.A_input = QTextEdit()
        self.A_input.setText('4 1 2\n1 3 2\n1 2 4')
        self.A_input.setStyleSheet("background-color: purple; color: white;")

        self.b_label = QLabel('Vector b (space-separated):')
        self.b_input = QLineEdit('4 5 6')
        self.b_input.setStyleSheet("background-color: purple; color: white;")

        self.x0_label = QLabel('Initial Guess x0 (space-separated):')
        self.x0_input = QLineEdit('0 0 0')
        self.x0_input.setStyleSheet("background-color: purple; color: white;")

        self.max_iterations_label = QLabel('Maximum Iterations:')
        self.max_iterations_input = QSpinBox()
        self.max_iterations_input.setValue(100)
        self.max_iterations_input.setStyleSheet("background-color: purple; color: white;")

        self.calculate_button = QPushButton('Calculate')
        self.calculate_button.clicked.connect(self.calculate)
        self.calculate_button.setStyleSheet("background-color: purple; color: white;")

        self.details_button = QPushButton('Details')
        self.details_button.clicked.connect(self.toggle_details)
        self.details_button.setStyleSheet("background-color: purple; color: white;")
        self.details_visible = False

        self.result_label = QLabel('')

        self.export_excel_button = QPushButton('Export to Excel')
        self.export_excel_button.clicked.connect(self.export_to_excel)
        self.export_excel_button.setStyleSheet("background-color: purple; color: white;")

        self.export_pdf_button = QPushButton('Export to PDF')
        self.export_pdf_button.clicked.connect(self.export_to_pdf)
        self.export_pdf_button.setStyleSheet("background-color: purple; color: white;")

        self.return_button = QPushButton('Return')
        self.return_button.clicked.connect(self.return_to_input)
        self.return_button.setStyleSheet("background-color: purple; color: white;")

        self.layout.addWidget(self.method_label)
        self.layout.addWidget(self.method_combo)
        self.layout.addWidget(self.A_label)
        self.layout.addWidget(self.A_input)
        self.layout.addWidget(self.b_label)
        self.layout.addWidget(self.b_input)
        self.layout.addWidget(self.x0_label)
        self.layout.addWidget(self.x0_input)
        self.layout.addWidget(self.max_iterations_label)
        self.layout.addWidget(self.max_iterations_input)
        self.layout.addWidget(self.calculate_button)
        self.layout.addWidget(self.details_button)
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.export_excel_button)
        self.layout.addWidget(self.export_pdf_button)
        self.layout.addWidget(self.return_button)

        self.setLayout(self.layout)

    def calculate(self):
        method = self.method_combo.currentText()
        A_input = self.A_input.toPlainText()
        b_input = self.b_input.text()
        x0_input = self.x0_input.text()
        max_iterations = self.max_iterations_input.value()

        try:
            A = np.array([list(map(float, row.split())) for row in A_input.split('\n')])
            b = np.array(list(map(float, b_input.split())))
            x0 = np.array(list(map(float, x0_input.split())))

            if method == 'Jacobi':
                solution, steps = jacobi(A, b, x0, max_iterations)
            else:
                solution, steps = gauss_seidel(A, b, x0, max_iterations)

            columns = ['Iteration'] + [f'x[{i}]' for i in range(len(b))] + ['Error']
            df = pd.DataFrame(steps, columns=columns)

            self.result_df = df
            self.solution = np.round(solution, 5)
            self.result_label.setText(f"Solution: {self.solution}")

        except Exception as e:
            QMessageBox.critical(self, 'Error', f"Error: {e}")

    def toggle_details(self):
        self.details_visible = not self.details_visible
        if self.details_visible:
            QMessageBox.information(self, 'Details', 'This program is written by Behran Saedi , Email: behransaedi@gmail.com')
        else:
            pass

    def export_to_excel(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'Excel Files (*.xlsx)')
        if file_path:
            excel_data = export_to_excel(self.result_df, self.method_combo.currentText())
            with open(file_path, 'wb') as f:
                f.write(excel_data.read())

    def export_to_pdf(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save File', '', 'PDF Files (*.pdf)')
        if file_path:
            pdf_data = export_to_pdf(self.result_df, self.method_combo.currentText())
            with open(file_path, 'wb') as f:
                f.write(pdf_data.read())

    def return_to_input(self):
        self.result_label.setText('')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NumericalMethodsApp()
    ex.show()
    sys.exit(app.exec())
