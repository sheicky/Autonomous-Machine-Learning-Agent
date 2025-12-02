import pandas as pd
import io

class DataManager:
    def __init__(self):
        self.df = None

    def load_data(self, uploaded_file):
        """
        Loads data from a Streamlit UploadedFile object.
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                self.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                self.df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                self.df = pd.read_parquet(uploaded_file)
            else:
                return False, "Unsupported file format. Please upload CSV, Excel, or Parquet."
            
            return True, "Data loaded successfully."
        except Exception as e:
            return False, f"Error loading data: {str(e)}"

    def get_summary(self):
        """
        Returns a dictionary containing summary statistics and metadata about the dataset.
        """
        if self.df is None:
            return None

        summary = {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "missing_values": self.df.isnull().sum().to_dict(),
            "total_missing": self.df.isnull().sum().sum(),
            "duplicate_rows": self.df.duplicated().sum(),
            "column_types": self.df.dtypes.astype(str).to_dict(),
            "numerical_columns": self.df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist(),
            "preview": self.df.head(),
            "description": self.df.describe(include='all').transpose() # Transpose for better view
        }
        return summary

    def get_dataframe(self):
        return self.df

